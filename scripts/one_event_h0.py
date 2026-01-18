"""H0 inference for a single GW event using dark siren method (MVP)."""

import argparse
import json
from pathlib import Path
import sys
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Import plotting helpers from plot_skymaps
scripts_dir = Path(__file__).parent
sys.path.insert(0, str(scripts_dir))
from plot_skymaps import (
    read_skymap_ligo, read_skymap_astropy,
    LIGO_SKYMAP_AVAILABLE, ASTROPY_AVAILABLE
)

# Import likelihood functions
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from epjc_h0.likelihood import (
    sky_prob_at, event_likelihood, normalize_posterior
)
from epjc_h0.cosmology import dL_std, dL_std_vectorized
from astropy_healpix import HEALPix

def load_manifest(manifest_path: Path) -> list:
    """Load manifest from JSON file."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, 'r') as f:
        return json.load(f)


def find_event_in_manifest(manifest: list, event_name: str) -> dict:
    """Find event entry in manifest."""
    for entry in manifest:
        if entry.get('event') == event_name:
            return entry
    raise ValueError(f"Event {event_name} not found in manifest")


def load_skymap(fits_path: Path):
    """Load skymap and return (prob_map, metadata, hp_obj)."""
    if LIGO_SKYMAP_AVAILABLE:
        prob, metadata = read_skymap_ligo(fits_path)
        hp_obj = None  # ligo.skymap handles this internally
    else:
        prob, metadata = read_skymap_astropy(fits_path)
        nside = metadata.get('nside', 512)
        nested = metadata.get('nest', False)
        hp_obj = HEALPix(nside=nside, order='nested' if nested else 'ring', frame='icrs')
    
    return prob, metadata, hp_obj


def create_sky_prob_func(prob_map: np.ndarray, metadata: dict, hp_obj) -> callable:
    """Create a function that returns p(Ω) given (ra_deg, dec_deg)."""
    def sky_prob(ra_deg: float, dec_deg: float) -> float:
        return sky_prob_at(ra_deg, dec_deg, prob_map, metadata, hp_obj)
    return sky_prob


def load_distance_pdf(pdf_path: Path, use_gw_likelihood: bool = True) -> tuple:
    """
    Load distance PDF from NPZ file.
    
    Parameters
    ----------
    pdf_path : Path
        Path to distance_pdf.npz file
    use_gw_likelihood : bool
        If True, use p_like (prior removed). If False, use p_post (posterior).
    
    Returns
    -------
    tuple
        (d_grid, p_grid) - distance grid and probability distribution
    """
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Distance PDF not found: {pdf_path}\n"
            f"Run: python scripts/plot_skymaps.py --distance --manifest data/skymaps/manifest.json"
        )
    
    data = np.load(pdf_path)
    
    # Check which keys are available
    if 'p_like' in data and 'p_post' in data:
        # New format with both posterior and likelihood
        if use_gw_likelihood:
            p_grid = data['p_like']
            dist_type = "likelihood (d^2 prior removed)"
        else:
            p_grid = data['p_post']
            dist_type = "posterior (includes d^2 prior)"
    elif 'pdf_grid' in data:
        # Old format (backward compatibility)
        p_grid = data['pdf_grid']
        dist_type = "posterior (legacy format, no prior removal available)"
        if use_gw_likelihood:
            print(f"  WARNING: Old format distance PDF, cannot remove prior. Using posterior.")
    else:
        raise ValueError(f"Unknown distance PDF format in {pdf_path}")
    
    d_grid = data['d_grid']
    
    # Verify normalization
    norm = np.trapezoid(p_grid, d_grid)
    print(f"  Loaded distance {dist_type}")
    print(f"    Normalization: {norm:.6f} (should be ~1.0)")
    
    if abs(norm - 1.0) > 0.01:
        print(f"    WARNING: Normalization is {norm:.6f}, renormalizing...")
        p_grid = p_grid / norm
    
    # Compute and print diagnostics
    d_mode_idx = np.argmax(p_grid)
    d_mode = d_grid[d_mode_idx]
    dx = d_grid[1] - d_grid[0] if len(d_grid) > 1 else 1.0
    cdf = np.cumsum(p_grid) * dx
    median_idx = np.searchsorted(cdf, 0.5)
    median_idx = min(median_idx, len(d_grid) - 1)
    d_median = d_grid[median_idx]
    
    print(f"    Mode: {d_mode:.1f} Mpc, Median: {d_median:.1f} Mpc")
    
    return d_grid, p_grid


def load_galaxies(galaxy_path: Path, z_min: float = 0.0, z_max: float = 10.0,
                 d_min: float = 0.0, d_max: float = 10000.0) -> np.ndarray:
    """
    Load galaxies from Parquet or CSV file.
    
    Parameters
    ----------
    galaxy_path : Path
        Path to galaxy catalog
    z_min, z_max : float
        Redshift range filter
    d_min, d_max : float
        Distance range filter (not used if z filtering is sufficient)
    
    Returns
    -------
    np.ndarray
        Structured array with fields: ra_deg, dec_deg, z, weight
    """
    if not galaxy_path.exists():
        raise FileNotFoundError(
            f"Galaxy catalog not found: {galaxy_path}\n"
            f"Please download GLADE+ or provide a catalog with columns: ra_deg, dec_deg, z, weight"
        )
    
    # Try parquet first, then CSV
    if galaxy_path.suffix == '.parquet':
        df = pd.read_parquet(galaxy_path)
    else:
        df = pd.read_csv(galaxy_path)
    
    # Check required columns
    required = ['ra_deg', 'dec_deg', 'z']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Add weight column if missing (default to 1.0)
    if 'weight' not in df.columns:
        df['weight'] = 1.0
    
    # Filter to finite values and redshift range
    df = df[
        df['ra_deg'].notna() & df['dec_deg'].notna() & df['z'].notna() &
        (df['z'] >= z_min) & (df['z'] <= z_max) &
        (df['z'] > 0)
    ].copy()
    
    if len(df) == 0:
        raise ValueError("No galaxies found after filtering")
    
    # Convert to structured array
    galaxies = np.array(
        list(zip(df['ra_deg'], df['dec_deg'], df['z'], df['weight'])),
        dtype=[('ra_deg', 'f8'), ('dec_deg', 'f8'), ('z', 'f8'), ('weight', 'f8')]
    )
    
    print(f"  Loaded {len(galaxies)} galaxies")
    return galaxies


def main():
    """Main function for H0 inference."""
    parser = argparse.ArgumentParser(
        description="H0 inference for a single GW event (MVP)"
    )
    parser.add_argument(
        '--event',
        type=str,
        required=True,
        help='Event name (must exist in manifest)'
    )
    parser.add_argument(
        '--h0-min',
        type=float,
        default=30.0,
        help='Minimum H0 value (default: 30)'
    )
    parser.add_argument(
        '--h0-max',
        type=float,
        default=150.0,
        help='Maximum H0 value (default: 150)'
    )
    parser.add_argument(
        '--h0-n',
        type=int,
        default=481,
        help='Number of H0 grid points (default: 481)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='both',
        choices=['std', 'ell', 'both'],
        help='Model type: std, ell, or both (default: both)'
    )
    parser.add_argument(
        '--kappa',
        type=float,
        default=1.0,
        help='kappa parameter for ell model (default: 1.0). Ignored if --scan-kappa is used.'
    )
    parser.add_argument(
        '--scan-kappa',
        action='store_true',
        help='Scan kappa values [0.0, 0.5, 1.0] for ELL model. Overrides --kappa.'
    )
    parser.add_argument(
        '--galaxies',
        type=str,
        default='data/glade_plus_min.parquet',
        help='Path to galaxy catalog (default: data/glade_plus_min.parquet)'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='data/skymaps/manifest.json',
        help='Path to manifest JSON (default: data/skymaps/manifest.json)'
    )
    parser.add_argument(
        '--out',
        type=str,
        default=None,
        help='Output figure path (default: figures/h0_post_<event>.png)'
    )
    parser.add_argument(
        '--use-gw-posterior',
        action='store_true',
        default=False,
        help='Use GW posterior directly (includes d^2 prior). Default: use GW likelihood (prior removed)'
    )
    args = parser.parse_args()
    
    # Set use_gw_likelihood based on flag (default True, False if --use-gw-posterior)
    args.use_gw_likelihood = not args.use_gw_posterior
    
    # Check availability
    if not ASTROPY_AVAILABLE:
        print("ERROR: astropy and astropy-healpix required")
        print("  Install with: pip install astropy astropy-healpix")
        return
    
    # Paths
    project_root = Path(__file__).parent.parent
    manifest_path = project_root / args.manifest
    galaxy_path = project_root / args.galaxies
    
    # Output path (deterministic filename)
    if args.out:
        out_path = Path(args.out)
    else:
        out_path = project_root / "figures" / f"h0_post_{args.event}.png"
    
    # Load manifest and find event
    print(f"Loading manifest: {manifest_path}", flush=True)
    manifest = load_manifest(manifest_path)
    entry = find_event_in_manifest(manifest, args.event)
    
    # Get FITS path
    fits_path_rel = entry.get('fits_path') or entry.get('extracted_fits_path')
    if not fits_path_rel:
        raise ValueError(f"No FITS path found for event {args.event}")
    
    fits_path = project_root / fits_path_rel
    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")
    
    # Get distance PDF path
    distance_pdf_path_rel = entry.get('distance_pdf_path')
    if not distance_pdf_path_rel:
        raise ValueError(
            f"No distance PDF found for event {args.event}\n"
            f"Run: python scripts/plot_skymaps.py --distance --manifest {args.manifest}"
        )
    
    distance_pdf_path = project_root / distance_pdf_path_rel
    
    print(f"\n{'='*60}", flush=True)
    print(f"H0 Inference for event: {args.event}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Step 1: Load skymap
    print(f"Step 1: Load skymap", flush=True)
    prob_map, metadata, hp_obj = load_skymap(fits_path)
    print(f"  Skymap loaded: nside={metadata.get('nside', 'unknown')}", flush=True)
    
    # Step 2: Load distance PDF
    print(f"Step 2: Load distance PDF", flush=True)
    d_grid, d_pdf = load_distance_pdf(distance_pdf_path, use_gw_likelihood=args.use_gw_likelihood)
    
    if not args.use_gw_likelihood:
        print(f"  WARNING: Using GW posterior directly (includes d^2 prior)", flush=True)
        print(f"    This may double-count priors and bias the H0 inference", flush=True)
    
    # Distance PDF diagnostics
    pdf_integral = np.trapezoid(d_pdf, d_grid)
    d_mode_idx = np.argmax(d_pdf)
    d_mode = d_grid[d_mode_idx]
    
    # Compute median via CDF
    dx = d_grid[1] - d_grid[0] if len(d_grid) > 1 else 1.0
    cdf = np.cumsum(d_pdf) * dx
    median_idx = np.searchsorted(cdf, 0.5)
    median_idx = min(median_idx, len(d_grid) - 1)
    d_median = d_grid[median_idx]
    
    print(f"  Distance PDF summary:", flush=True)
    print(f"    Grid: {len(d_grid)} points, range [{d_grid.min():.1f}, {d_grid.max():.1f}] Mpc", flush=True)
    print(f"    PDF integral: {pdf_integral:.6f} (should be ~1.0)", flush=True)
    print(f"    Distance mode: {d_mode:.1f} Mpc", flush=True)
    print(f"    Distance median: {d_median:.1f} Mpc", flush=True)
    
    if abs(pdf_integral - 1.0) > 0.01:
        print(f"    WARNING: PDF normalization is {pdf_integral:.6f}, expected ~1.0", flush=True)
    
    # Compute 99% credible interval for distance window filter
    p05_idx = np.searchsorted(cdf, 0.005)
    p95_idx = np.searchsorted(cdf, 0.995)
    p05_idx = max(0, p05_idx - 1)
    p95_idx = min(len(d_grid) - 1, p95_idx)
    d_lo = d_grid[p05_idx]
    d_hi = d_grid[p95_idx]
    print(f"    99% credible interval: [{d_lo:.1f}, {d_hi:.1f}] Mpc", flush=True)
    
    # Step 3: Load galaxies
    print(f"Step 3: Load galaxies", flush=True)
    galaxies_all = load_galaxies(galaxy_path)
    print(f"  Total galaxies loaded: {len(galaxies_all):,}", flush=True)
    
    # Step 3.5: Apply distance window filter (MVP)
    print(f"Step 3.5: Apply distance window filter (MVP)", flush=True)
    print(f"  Computing d_L for {len(galaxies_all):,} galaxies at H0=70...", flush=True)
    H0_ref = 70.0  # Reference H0 for distance window
    z_array = galaxies_all['z']
    
    # Process in chunks to show progress and handle memory
    chunk_size = 500000  # Smaller chunks to reduce memory pressure
    n_chunks = (len(z_array) + chunk_size - 1) // chunk_size
    d_L_ref = np.zeros(len(z_array), dtype=np.float32)  # Use float32 to save memory
    
    import gc
    
    try:
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(z_array))
            print(f"    Processing chunk {chunk_idx+1}/{n_chunks} (galaxies {start_idx:,} to {end_idx:,})...", flush=True)
            sys.stdout.flush()
            
            try:
                chunk_z = z_array[start_idx:end_idx]
                print(f"      Computing d_L for {len(chunk_z):,} galaxies...", flush=True)
                sys.stdout.flush()
                
                # Add timeout check - if chunk takes too long, report
                import time
                chunk_start = time.time()
                
                chunk_result = dL_std_vectorized(chunk_z, H0_ref, Om=0.3)
                
                chunk_time = time.time() - chunk_start
                print(f"      d_L computation took {chunk_time:.1f}s", flush=True)
                sys.stdout.flush()
                
                d_L_ref[start_idx:end_idx] = chunk_result.astype(np.float32)
                
                print(f"      ✓ Chunk {chunk_idx+1} complete", flush=True)
                sys.stdout.flush()
                
                # Free memory
                del chunk_z, chunk_result
                gc.collect()
                
                # Force output flush
                sys.stdout.flush()
                sys.stderr.flush()
                
            except MemoryError as e:
                print(f"      ERROR: Out of memory in chunk {chunk_idx+1}: {e}", flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                traceback.print_exc(file=sys.stdout)
                traceback.print_exc(file=sys.stderr)
                sys.stdout.flush()
                sys.stderr.flush()
                raise
            except KeyboardInterrupt:
                print(f"      INTERRUPTED in chunk {chunk_idx+1}", flush=True)
                sys.stdout.flush()
                raise
            except Exception as e:
                print(f"      ERROR in chunk {chunk_idx+1}: {type(e).__name__}: {e}", flush=True)
                sys.stdout.flush()
                sys.stderr.flush()
                traceback.print_exc(file=sys.stdout)
                traceback.print_exc(file=sys.stderr)
                sys.stdout.flush()
                sys.stderr.flush()
                raise
        
        print(f"  Filtering galaxies by distance window...", flush=True)
        sys.stdout.flush()
        mask_distance = (d_L_ref >= d_lo) & (d_L_ref <= d_hi)
        galaxies = galaxies_all[mask_distance]
        del d_L_ref  # Free memory
        print(f"  Galaxies after distance window filter: {len(galaxies):,} ({100*len(galaxies)/len(galaxies_all):.1f}%)", flush=True)
    except Exception as e:
        print(f"  FATAL ERROR in distance window filter: {e}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        # Fallback: use all galaxies
        print(f"  Falling back to all galaxies (filter disabled)", flush=True)
        galaxies = galaxies_all
    
    if len(galaxies) == 0:
        print(f"  WARNING: No galaxies remain after distance window filter!", flush=True)
        print(f"    Consider widening the filter or checking distance PDF units", flush=True)
        # Fallback: use all galaxies
        galaxies = galaxies_all
        print(f"    Using all {len(galaxies):,} galaxies (filter disabled)", flush=True)
    
    # Check distance PDF semantics
    print(f"\nDistance PDF consistency check:", flush=True)
    # Compute typical galaxy distances at H0=70
    d_L_typical = dL_std_vectorized(galaxies['z'], 70.0, Om=0.3)
    d_L_min_gal = np.min(d_L_typical)
    d_L_max_gal = np.max(d_L_typical)
    d_L_median_gal = np.median(d_L_typical)
    
    print(f"  Galaxy d_L range (H0=70): [{d_L_min_gal:.1f}, {d_L_median_gal:.1f}, {d_L_max_gal:.1f}] Mpc", flush=True)
    print(f"  Distance PDF range: [{d_grid.min():.1f}, {d_median:.1f}, {d_grid.max():.1f}] Mpc", flush=True)
    
    # Check if PDF peak is far from galaxy range
    if d_mode < d_L_min_gal * 0.5 or d_mode > d_L_max_gal * 2.0:
        print(f"  WARNING: Distance PDF mode ({d_mode:.1f} Mpc) is far from galaxy d_L range!", flush=True)
        print(f"    This may indicate unit mismatch (comoving vs luminosity distance)", flush=True)
    if d_median < d_L_min_gal * 0.5 or d_median > d_L_max_gal * 2.0:
        print(f"  WARNING: Distance PDF median ({d_median:.1f} Mpc) is far from galaxy d_L range!", flush=True)
    
    # Step 4: Create H0 grid
    print(f"Step 4: Create H0 grid", flush=True)
    H0_grid = np.linspace(args.h0_min, args.h0_max, args.h0_n)
    print(f"  H0 grid: [{args.h0_min}, {args.h0_max}] with {args.h0_n} points", flush=True)
    
    # Diagnostic: Compute dL for reference H0 values
    print(f"\nDiagnostic: Galaxy distance ranges at reference H0 values", flush=True)
    for H0_ref in [60.0, 100.0]:
        d_L_array = dL_std_vectorized(galaxies['z'], H0_ref, Om=0.3)
        print(f"  H0={H0_ref:.0f} km/s/Mpc:", flush=True)
        print(f"    d_L range: [{d_L_array.min():.1f}, {np.median(d_L_array):.1f}, {d_L_array.max():.1f}] Mpc", flush=True)
    
    # Diagnostic: Compute ddL/dz for reference H0 values (Jacobian check)
    print(f"\nDiagnostic: Jacobian ddL/dz at reference H0 values", flush=True)
    from epjc_h0.cosmology import ddL_dz_std_vectorized
    for H0_ref in [60.0, 120.0]:
        ddL_dz_array = ddL_dz_std_vectorized(galaxies['z'], H0_ref, Om=0.3)
        print(f"  H0={H0_ref:.0f} km/s/Mpc:", flush=True)
        print(f"    ddL/dz median: {np.median(ddL_dz_array):.2f} Mpc", flush=True)
        print(f"    ddL/dz range: [{ddL_dz_array.min():.2f}, {ddL_dz_array.max():.2f}] Mpc", flush=True)
        # Check scaling: should scale roughly like 1/H0
        if H0_ref == 60.0:
            ddL_dz_60 = np.median(ddL_dz_array)
        elif H0_ref == 120.0:
            ddL_dz_120 = np.median(ddL_dz_array)
            ratio_expected = 60.0 / 120.0
            ratio_actual = ddL_dz_60 / ddL_dz_120
            print(f"    Scaling check: ddL/dz(60)/ddL/dz(120) = {ratio_actual:.3f} (expected ~{ratio_expected:.3f})", flush=True)
    
    # Step 5: Compute likelihoods
    # Pass (prob_map, metadata, hp_obj) tuple for vectorized processing
    sky_prob_data = (prob_map, metadata, hp_obj)
    
    print(f"Step 5: Compute likelihoods", flush=True)
    results = {}
    
    # Determine distance_pdf_kind based on use_gw_likelihood flag
    if args.use_gw_likelihood:
        distance_pdf_kind = "likelihood"
        print(f"  Using distance PDF as GW likelihood (prior removed)", flush=True)
    else:
        distance_pdf_kind = "posterior_d2"
        print(f"  Using distance PDF as GW posterior (will remove d^2 prior)", flush=True)
    
    if args.model in ['std', 'both']:
        print(f"  Computing std model...", flush=True)
        likelihood_std = event_likelihood(
            H0_grid, galaxies, sky_prob_data, d_grid, d_pdf,
            model='std', Om=0.3,
            distance_pdf_kind=distance_pdf_kind, d_min=1.0
        )
        posterior_std, norm_std = normalize_posterior(likelihood_std, H0_grid)
        results['std'] = (posterior_std, norm_std, likelihood_std)
        print(f"    Normalization: {norm_std:.2e}")
        
        # Likelihood slope check
        print(f"    Likelihood slope check:")
        print(f"      L(H0_min={args.h0_min:.1f}) = {likelihood_std[0]:.2e}")
        mid_idx = len(H0_grid) // 2
        print(f"      L(H0_mid={H0_grid[mid_idx]:.1f}) = {likelihood_std[mid_idx]:.2e}")
        print(f"      L(H0_max={args.h0_max:.1f}) = {likelihood_std[-1]:.2e}")
        
        # Find MAP
        map_idx = np.argmax(posterior_std)
        H0_map = H0_grid[map_idx]
        is_boundary = (map_idx == 0) or (map_idx == len(H0_grid) - 1)
        print(f"      MAP H0 = {H0_map:.1f} km/s/Mpc {'(BOUNDARY!)' if is_boundary else ''}")
    
    if args.model in ['ell', 'both']:
        # Determine kappa values to scan
        if args.scan_kappa:
            kappa_values = [0.0, 0.5, 1.0]
            print(f"  Scanning ell model with kappa values: {kappa_values}", flush=True)
        else:
            kappa_values = [args.kappa]
            print(f"  Computing ell model (kappa={args.kappa})...", flush=True)
        
        # Compute likelihoods for each kappa value
        for kappa in kappa_values:
            print(f"    Computing kappa={kappa}...", flush=True)
            likelihood_ell = event_likelihood(
                H0_grid, galaxies, sky_prob_data, d_grid, d_pdf,
                model='ell', kappa=kappa, Om=0.3,
                distance_pdf_kind=distance_pdf_kind, d_min=1.0
            )
            posterior_ell, norm_ell = normalize_posterior(likelihood_ell, H0_grid)
            
            # Store results with kappa as key
            key = f'ell_kappa_{kappa}'
            results[key] = (posterior_ell, norm_ell, likelihood_ell, kappa)
            
            print(f"      Normalization: {norm_ell:.2e}", flush=True)
            
            # Likelihood slope check
            print(f"      Likelihood slope check:", flush=True)
            print(f"        L(H0_min={args.h0_min:.1f}) = {likelihood_ell[0]:.2e}", flush=True)
            mid_idx = len(H0_grid) // 2
            print(f"        L(H0_mid={H0_grid[mid_idx]:.1f}) = {likelihood_ell[mid_idx]:.2e}", flush=True)
            print(f"        L(H0_max={args.h0_max:.1f}) = {likelihood_ell[-1]:.2e}", flush=True)
            
            # Find MAP
            map_idx = np.argmax(posterior_ell)
            H0_map = H0_grid[map_idx]
            is_boundary = (map_idx == 0) or (map_idx == len(H0_grid) - 1)
            print(f"        MAP H0 = {H0_map:.1f} km/s/Mpc {'(BOUNDARY!)' if is_boundary else ''}", flush=True)
    
    # Step 6: Plot
    print(f"Step 6: Plot posterior")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'std' in results:
        posterior_std, _, likelihood_std = results['std']
        ax.plot(H0_grid, posterior_std, 'b-', linewidth=2, label='Standard LCDM', alpha=0.8)
        ax.fill_between(H0_grid, posterior_std, alpha=0.3, color='blue')
        
        # Add vertical line at MAP
        map_idx = np.argmax(posterior_std)
        H0_map_std = H0_grid[map_idx]
        is_boundary_std = (map_idx == 0) or (map_idx == len(H0_grid) - 1)
        ax.axvline(H0_map_std, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
        label_map = f'MAP={H0_map_std:.1f}'
        if is_boundary_std:
            label_map += ' (boundary)'
        y_max = ax.get_ylim()[1]
        ax.text(H0_map_std, y_max * 0.9, label_map, 
               rotation=90, verticalalignment='top', fontsize=9, color='blue',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Plot all ELL models (with different kappa values)
    ell_keys = [k for k in results.keys() if k.startswith('ell_kappa_')]
    if ell_keys:
        # Sort by kappa value
        ell_keys_sorted = sorted(ell_keys, key=lambda k: float(k.split('_')[-1]))
        
        # Colors and styles for different kappa values
        colors = ['red', 'orange', 'purple']
        linestyles = ['--', '-.', ':']
        
        for idx, key in enumerate(ell_keys_sorted):
            posterior_ell, _, likelihood_ell, kappa = results[key]
            color = colors[idx % len(colors)]
            linestyle = linestyles[idx % len(linestyles)]
            
            # Special label for kappa=0 (should match STD)
            if kappa == 0.0:
                label = f'ELL (κ={kappa:.1f}, should match STD)'
            else:
                label = f'ELL (κ={kappa:.1f})'
            
            ax.plot(H0_grid, posterior_ell, color=color, linestyle=linestyle, 
                   linewidth=2, label=label, alpha=0.8)
            ax.fill_between(H0_grid, posterior_ell, alpha=0.15, color=color)
            
            # Add vertical line at MAP
            map_idx = np.argmax(posterior_ell)
            H0_map_ell = H0_grid[map_idx]
            is_boundary_ell = (map_idx == 0) or (map_idx == len(H0_grid) - 1)
            ax.axvline(H0_map_ell, color=color, linestyle=linestyle, 
                      linewidth=1.5, alpha=0.7)
            label_map = f'MAP={H0_map_ell:.1f}'
            if is_boundary_ell:
                label_map += ' (boundary)'
            y_max = ax.get_ylim()[1]
            y_pos = y_max * (0.8 - idx * 0.1)  # Stagger vertical lines
            ax.text(H0_map_ell, y_pos, label_map, 
                   rotation=90, verticalalignment='top', fontsize=9, color=color,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('H₀ (km/s/Mpc)', fontsize=12)
    ax.set_ylabel('Posterior P(H₀)', fontsize=12)
    ax.set_title(f'H₀ Posterior: {args.event}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add note about MVP limitations
    ax.text(0.02, 0.98, 
           'MVP: Ignores selection effects\nand out-of-catalog term',
           transform=ax.transAxes, fontsize=9,
           verticalalignment='top', bbox=dict(boxstyle='round', 
           facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Plot saved: {out_path}")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"Summary Statistics")
    print(f"{'='*60}")
    
    def compute_quantiles(posterior, grid):
        """Compute median and 16th/84th percentiles."""
        dx = grid[1] - grid[0] if len(grid) > 1 else 1.0
        # Normalize posterior to ensure it integrates to 1
        norm = np.trapz(posterior, grid)
        if norm > 0:
            posterior_norm = posterior / norm
        else:
            posterior_norm = posterior
        # Compute CDF
        cdf = np.cumsum(posterior_norm) * dx
        # Find quantiles
        median_idx = np.searchsorted(cdf, 0.5)
        p16_idx = np.searchsorted(cdf, 0.16)
        p84_idx = np.searchsorted(cdf, 0.84)
        # Clamp indices
        median_idx = min(median_idx, len(grid) - 1)
        p16_idx = min(p16_idx, len(grid) - 1)
        p84_idx = min(p84_idx, len(grid) - 1)
        return grid[median_idx], grid[p16_idx], grid[p84_idx]
    
    if 'std' in results:
        posterior_std, _, _ = results['std']
        H0_median_std, H0_16_std, H0_84_std = compute_quantiles(posterior_std, H0_grid)
        map_idx = np.argmax(posterior_std)
        H0_map_std = H0_grid[map_idx]
        print(f"Standard LCDM:")
        print(f"  H₀ (median) = {H0_median_std:.1f} +{H0_84_std - H0_median_std:.1f} -{H0_median_std - H0_16_std:.1f} km/s/Mpc")
        print(f"  H₀ (MAP) = {H0_map_std:.1f} km/s/Mpc")
    
    # Print summary for all ELL models
    ell_keys = [k for k in results.keys() if k.startswith('ell_kappa_')]
    if ell_keys:
        ell_keys_sorted = sorted(ell_keys, key=lambda k: float(k.split('_')[-1]))
        for key in ell_keys_sorted:
            posterior_ell, _, _, kappa = results[key]
            H0_median_ell, H0_16_ell, H0_84_ell = compute_quantiles(posterior_ell, H0_grid)
            map_idx = np.argmax(posterior_ell)
            H0_map_ell = H0_grid[map_idx]
            if kappa == 0.0:
                print(f"ELL (κ={kappa:.1f}, should match STD):")
            else:
                print(f"ELL (κ={kappa:.1f}):")
            print(f"  H₀ (median) = {H0_median_ell:.1f} +{H0_84_ell - H0_median_ell:.1f} -{H0_median_ell - H0_16_ell:.1f} km/s/Mpc")
            print(f"  H₀ (MAP) = {H0_map_ell:.1f} km/s/Mpc")
    
    print(f"\nDone!")


if __name__ == "__main__":
    # Open log file for writing
    log_file_path = Path(__file__).parent.parent / "logs_h0_inference.txt"
    
    # Simple approach: just run main and let it print to console
    # User can redirect if needed: python script.py > log.txt 2>&1
    try:
        main()
        print(f"\n{'='*60}", flush=True)
        print(f"SUCCESS: Script completed", flush=True)
        print(f"{'='*60}", flush=True)
    except KeyboardInterrupt:
        print(f"\n{'='*60}", flush=True)
        print(f"INTERRUPTED by user", flush=True)
        print(f"{'='*60}", flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}", flush=True)
        print(f"FATAL ERROR: {e}", flush=True)
        print(f"{'='*60}", flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
        sys.exit(1)

"""Depth scan analysis: H0 posterior vs catalog depth (zmax) for 3D skymap mode."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    event_likelihood, normalize_posterior
)
from epjc_h0.cosmology import dL_std_vectorized
from epjc_h0.results_metrics import (
    posterior_mass_in_window, posterior_quantiles,
    CMB_WINDOW_LO, CMB_WINDOW_HI, LOCAL_WINDOW_LO, LOCAL_WINDOW_HI
)
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


def load_3d_skymap(fits_path: Path):
    """
    Load 3D skymap with distance information.
    
    Returns
    -------
    tuple
        (prob_map, distmu_map, distsigma_map, distnorm_map, metadata, hp_obj)
    """
    from astropy.io import fits
    
    with fits.open(str(fits_path)) as hdul:
        hdu = hdul[1]
        
        # Check for 3D distance information
        if 'DISTMU' not in hdu.data.dtype.names:
            raise ValueError(f"Distance information (DISTMU) not found in FITS file: {fits_path}")
        
        # Extract arrays - handle both regular HEALPix and UNIQ formats
        if 'UNIQ' in hdu.data.dtype.names:
            # UNIQ format: need to convert to regular HEALPix
            prob_raw = hdu.data['PROBDENSITY']
            distmu_raw = hdu.data['DISTMU']
            distsigma_raw = hdu.data['DISTSIGMA']
            distnorm_raw = hdu.data['DISTNORM']
            uniq = hdu.data['UNIQ']
            
            # Convert UNIQ to regular HEALPix grid (target nside=512)
            target_nside = 512
            hp_target = HEALPix(nside=target_nside, order='nested', frame='icrs')
            npix_target = hp_target.npix
            
            prob_map = np.zeros(npix_target)
            distmu_map = np.zeros(npix_target)
            distsigma_map = np.zeros(npix_target)
            distnorm_map = np.zeros(npix_target)
            
            # Decode UNIQ and accumulate values
            for i in range(len(uniq)):
                u = uniq[i]
                p = prob_raw[i] if i < len(prob_raw) else 0.0
                mu = distmu_raw[i] if i < len(distmu_raw) else 0.0
                sigma = distsigma_raw[i] if i < len(distsigma_raw) else 0.0
                norm = distnorm_raw[i] if i < len(distnorm_raw) else 0.0
                
                # Decode UNIQ
                level = 0
                while 4 * (4 ** (level + 1)) <= u:
                    level += 1
                nside_u = 2 ** level
                ipix_u = u - 4 * (4 ** level)
                
                # Get pixel center coordinates
                hp_source = HEALPix(nside=nside_u, order='nested', frame='icrs')
                lon_u, lat_u = hp_source.healpix_to_lonlat([ipix_u])
                
                # Find target pixel
                ipix_target = hp_target.lonlat_to_healpix(lon_u, lat_u)[0]
                
                # Accumulate values
                if ipix_target < npix_target:
                    prob_map[ipix_target] = max(prob_map[ipix_target], p)
                    if distmu_map[ipix_target] == 0 or mu > 0:
                        distmu_map[ipix_target] = mu if mu > 0 else distmu_map[ipix_target]
                        distsigma_map[ipix_target] = sigma if sigma > 0 else distsigma_map[ipix_target]
                        distnorm_map[ipix_target] = norm if norm > 0 else distnorm_map[ipix_target]
            
            metadata = {'nside': target_nside, 'nest': True}
            hp_obj = HEALPix(nside=target_nside, order='nested', frame='icrs')
        else:
            # Regular HEALPix format
            prob_map = hdu.data['PROBDENSITY']
            distmu_map = hdu.data['DISTMU']
            distsigma_map = hdu.data['DISTSIGMA']
            distnorm_map = hdu.data['DISTNORM']
            
            # Get metadata
            if LIGO_SKYMAP_AVAILABLE:
                from plot_skymaps import read_skymap_ligo
                prob_temp, metadata = read_skymap_ligo(fits_path)
                hp_obj = None
            else:
                nside = hdu.header.get('NSIDE', 512)
                nested = hdu.header.get('NESTED', False)
                metadata = {'nside': nside, 'nest': bool(nested)}
                hp_obj = HEALPix(nside=nside, order='nested' if nested else 'ring', frame='icrs')
    
    return prob_map, distmu_map, distsigma_map, distnorm_map, metadata, hp_obj


def load_galaxies(galaxy_path: Path, z_max: float = 10.0) -> np.ndarray:
    """
    Load galaxies from Parquet or CSV file and filter by z <= z_max.
    
    Parameters
    ----------
    galaxy_path : Path
        Path to galaxy catalog
    z_max : float
        Maximum redshift to include
    
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
        (df['z'] > 0) & (df['z'] <= z_max)
    ].copy()
    
    # Convert to structured array
    galaxies = np.zeros(len(df), dtype=[
        ('ra_deg', 'f8'),
        ('dec_deg', 'f8'),
        ('z', 'f8'),
        ('weight', 'f8')
    ])
    
    galaxies['ra_deg'] = df['ra_deg'].values
    galaxies['dec_deg'] = df['dec_deg'].values
    galaxies['z'] = df['z'].values
    galaxies['weight'] = df['weight'].values if 'weight' in df.columns else 1.0
    
    return galaxies


def compute_quantiles(posterior: np.ndarray, grid: np.ndarray):
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


def run_depth_scan(event_id: str, zmax_list: list, model_list=None, 
                   H0_min: float = 20.0, H0_max: float = 120.0, H0_n: int = 201,
                   galaxy_path: Path = None, manifest_path: Path = None,
                   project_root: Path = None) -> pd.DataFrame:
    """
    Run depth scan analysis: compute H0 posterior for different catalog depths (zmax).
    
    Parameters
    ----------
    event_id : str
        Event identifier
    zmax_list : list
        List of maximum redshifts to scan
    model_list : list, optional
        List of (model_name, kappa) tuples, e.g. [("std", 0.0), ("ell", 1.0)]
        Default: [("std", 0.0), ("ell", 1.0)]
    H0_min, H0_max, H0_n : float, float, int
        H0 grid parameters
    galaxy_path : Path
        Path to galaxy catalog
    manifest_path : Path
        Path to manifest JSON
    project_root : Path
        Project root directory
    
    Returns
    -------
    pd.DataFrame
        Results table with columns: zmax, model, kappa, H0_MAP, H0_median, H0_p16, H0_p84, n_galaxies
    """
    if model_list is None:
        model_list = [("std", 0.0), ("ell", 1.0)]
    
    if project_root is None:
        project_root = Path(__file__).parent.parent
    
    if manifest_path is None:
        manifest_path = project_root / "data/skymaps/manifest.json"
    
    if galaxy_path is None:
        galaxy_path = project_root / "data/glade_plus_min.parquet"
    
    print(f"\n{'='*60}", flush=True)
    print(f"Depth Scan Analysis: {event_id}", flush=True)
    print(f"{'='*60}", flush=True)
    
    # Load manifest and find event
    print(f"Loading manifest: {manifest_path}", flush=True)
    manifest = load_manifest(manifest_path)
    entry = find_event_in_manifest(manifest, event_id)
    
    # Get FITS path
    fits_path_rel = entry.get('fits_path') or entry.get('extracted_fits_path')
    if not fits_path_rel:
        raise ValueError(f"No FITS path found for event {event_id}")
    
    fits_path = project_root / fits_path_rel
    if not fits_path.exists():
        raise FileNotFoundError(f"FITS file not found: {fits_path}")
    
    # Load 3D skymap
    print(f"Loading 3D skymap...", flush=True)
    prob_map, distmu_map, distsigma_map, distnorm_map, metadata, hp_obj = load_3d_skymap(fits_path)
    print(f"  3D skymap loaded: nside={metadata.get('nside', 'unknown')}", flush=True)
    
    # Prepare 3D skymap data tuple
    sky_prob_data_3d = (prob_map, distmu_map, distsigma_map, distnorm_map, metadata, hp_obj)
    
    # Create H0 grid
    H0_grid = np.linspace(H0_min, H0_max, H0_n)
    print(f"H0 grid: [{H0_min}, {H0_max}] with {H0_n} points", flush=True)
    
    # Dummy d_grid/d_pdf (not used in 3D mode, but required by function signature)
    d_grid = np.linspace(1.0, 5000.0, 1000)
    d_pdf = np.ones_like(d_grid) / np.trapz(np.ones_like(d_grid), d_grid)
    
    # Store results
    results = []
    
    # Store posteriors for each zmax and model
    posteriors_std_3d = []
    posteriors_ell_3d = []
    valid_zmax_indices = []
    
    # Scan over zmax values
    print(f"\nScanning {len(zmax_list)} zmax values: {zmax_list}", flush=True)
    for zmax_idx, zmax in enumerate(zmax_list):
        print(f"\n{'='*60}", flush=True)
        print(f"zmax = {zmax:.3f} ({zmax_idx+1}/{len(zmax_list)})", flush=True)
        print(f"{'='*60}", flush=True)
        
        # Load galaxies filtered by zmax
        print(f"Loading galaxies with z <= {zmax}...", flush=True)
        galaxies = load_galaxies(galaxy_path, z_max=zmax)
        n_galaxies = len(galaxies)
        print(f"  Loaded {n_galaxies:,} galaxies", flush=True)
        
        if n_galaxies == 0:
            print(f"  WARNING: No galaxies found for zmax={zmax}, skipping", flush=True)
            continue
        
        # Compute H0 posterior for each model
        for model_name, kappa in model_list:
            print(f"\n  Computing {model_name} model (kappa={kappa})...", flush=True)
            
            try:
                # Compute likelihood
                likelihood = event_likelihood(
                    H0_grid, galaxies, sky_prob_data_3d, d_grid, d_pdf,
                    model=model_name, kappa=kappa, Om=0.3,
                    distance_pdf_kind="likelihood", d_min=1.0,
                    use_3d_skymap=True
                )
                
                # Normalize to posterior
                posterior, norm = normalize_posterior(likelihood, H0_grid)
                
                # Compute quantiles using new function
                quantiles = posterior_quantiles(H0_grid, posterior, qs=(0.16, 0.5, 0.84))
                H0_p16, H0_median, H0_p84 = quantiles
                
                # Find MAP
                map_idx = np.argmax(posterior)
                H0_map = H0_grid[map_idx]
                is_boundary = (map_idx == 0) or (map_idx == len(H0_grid) - 1)
                
                # Compute probability masses in CMB and Local windows
                P_CMB = posterior_mass_in_window(H0_grid, posterior, CMB_WINDOW_LO, CMB_WINDOW_HI)
                P_Local = posterior_mass_in_window(H0_grid, posterior, LOCAL_WINDOW_LO, LOCAL_WINDOW_HI)
                
                # Compute ratio R = P_CMB / P_Local (with regularization)
                R = P_CMB / max(P_Local, 1e-12)
                DeltaP = P_CMB - P_Local
                
                print(f"    MAP H0 = {H0_map:.1f} km/s/Mpc {'(BOUNDARY!)' if is_boundary else ''}", flush=True)
                print(f"    Median H0 = {H0_median:.1f} km/s/Mpc", flush=True)
                print(f"    16-84% CI: [{H0_p16:.1f}, {H0_p84:.1f}] km/s/Mpc", flush=True)
                print(f"    P[CMB 65-70] = {P_CMB:.4f}, P[Local 70-75] = {P_Local:.4f}, R = {R:.4f}", flush=True)
                
                # Store result
                results.append({
                    'zmax': zmax,
                    'model': model_name,
                    'kappa': kappa,
                    'H0_MAP': H0_map,
                    'H0_median': H0_median,
                    'H0_p16': H0_p16,
                    'H0_p84': H0_p84,
                    'P_CMB': P_CMB,
                    'P_Local': P_Local,
                    'R': R,
                    'DeltaP': DeltaP,
                    'n_galaxies': n_galaxies,
                    'is_boundary': is_boundary
                })
                
                # Store posterior for saving
                if model_name == 'std':
                    posteriors_std_3d.append(posterior)
                    if zmax_idx not in valid_zmax_indices:
                        valid_zmax_indices.append(zmax_idx)
                elif model_name == 'ell':
                    posteriors_ell_3d.append(posterior)
                
            except Exception as e:
                print(f"    ERROR: Failed to compute {model_name} model: {e}", flush=True)
                import traceback
                traceback.print_exc()
                continue
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Prepare posteriors arrays for saving
    if len(posteriors_std_3d) > 0 and len(posteriors_ell_3d) > 0:
        # Ensure same number of zmax values
        n_zmax = min(len(posteriors_std_3d), len(posteriors_ell_3d))
        posteriors_std_3d_array = np.array(posteriors_std_3d[:n_zmax])
        posteriors_ell_3d_array = np.array(posteriors_ell_3d[:n_zmax])
        zmax_list_valid = [zmax_list[i] for i in valid_zmax_indices[:n_zmax]]
    else:
        posteriors_std_3d_array = None
        posteriors_ell_3d_array = None
        zmax_list_valid = None
    
    return df_results, H0_grid, posteriors_std_3d_array, posteriors_ell_3d_array, zmax_list_valid


def plot_depth_scan(df_results: pd.DataFrame, event_id: str, out_path: Path):
    """
    Plot depth scan results.
    
    Parameters
    ----------
    df_results : pd.DataFrame
        Results from run_depth_scan
    event_id : str
        Event identifier
    out_path : Path
        Output figure path
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: H0_MAP vs zmax for STD and ELL
    ax1 = axes[0]
    
    # Filter results for STD and ELL models
    df_std = df_results[df_results['model'] == 'std'].copy()
    df_ell = df_results[df_results['model'] == 'ell'].copy()
    
    if len(df_std) > 0:
        df_std = df_std.sort_values('zmax')
        ax1.plot(df_std['zmax'], df_std['H0_MAP'], 'b-o', linewidth=2, 
                markersize=6, label='STD 3D', alpha=0.8)
        # Add error bars (16-84% CI)
        ax1.fill_between(df_std['zmax'], df_std['H0_p16'], df_std['H0_p84'],
                        alpha=0.2, color='blue')
    
    if len(df_ell) > 0:
        df_ell = df_ell.sort_values('zmax')
        ax1.plot(df_ell['zmax'], df_ell['H0_MAP'], 'r--s', linewidth=2,
                markersize=6, label='ELL 3D', alpha=0.8)
        # Add error bars (16-84% CI)
        ax1.fill_between(df_ell['zmax'], df_ell['H0_p16'], df_ell['H0_p84'],
                        alpha=0.2, color='red')
    
    ax1.set_xlabel('Catalog depth zmax', fontsize=12)
    ax1.set_ylabel('H₀ MAP (km/s/Mpc)', fontsize=12)
    ax1.set_title(f'H₀ MAP vs Catalog Depth: {event_id}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Plot 2: DeltaH0 = H0_ELL - H0_STD vs zmax
    ax2 = axes[1]
    
    if len(df_std) > 0 and len(df_ell) > 0:
        # Merge on zmax
        df_merged = pd.merge(df_std[['zmax', 'H0_MAP', 'H0_median']], 
                            df_ell[['zmax', 'H0_MAP', 'H0_median']], 
                            on='zmax', suffixes=('_std', '_ell'))
        df_merged = df_merged.sort_values('zmax')
        
        # Compute DeltaH0
        df_merged['DeltaH0_MAP'] = df_merged['H0_MAP_ell'] - df_merged['H0_MAP_std']
        df_merged['DeltaH0_median'] = df_merged['H0_median_ell'] - df_merged['H0_median_std']
        
        ax2.plot(df_merged['zmax'], df_merged['DeltaH0_MAP'], 'g-o', 
                linewidth=2, markersize=6, label='ΔH₀ (MAP)', alpha=0.8)
        ax2.plot(df_merged['zmax'], df_merged['DeltaH0_median'], 'g--s',
                linewidth=2, markersize=6, label='ΔH₀ (median)', alpha=0.8)
        
        # Add horizontal line at zero
        ax2.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Catalog depth zmax', fontsize=12)
    ax2.set_ylabel('ΔH₀ = H₀(ELL) - H₀(STD) (km/s/Mpc)', fontsize=12)
    ax2.set_title(f'H₀ Drift vs Catalog Depth: {event_id}', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved: {out_path}", flush=True)


def plot_prob_mass_windows(df_metrics: pd.DataFrame, event_id: str, project_root: Path):
    """Plot probability mass in CMB and Local windows vs zmax."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_metrics = df_metrics.sort_values('zmax')
    
    # Plot STD
    ax.plot(df_metrics['zmax'], df_metrics['P_CMB_std'], 'b-o', 
           linewidth=2, markersize=6, label='STD 3D: P[65-70]', alpha=0.8)
    ax.plot(df_metrics['zmax'], df_metrics['P_Local_std'], 'b--s',
           linewidth=2, markersize=6, label='STD 3D: P[70-75]', alpha=0.8)
    
    # Plot ELL
    ax.plot(df_metrics['zmax'], df_metrics['P_CMB_ell'], 'r-o',
           linewidth=2, markersize=6, label='ELL 3D: P[65-70]', alpha=0.8)
    ax.plot(df_metrics['zmax'], df_metrics['P_Local_ell'], 'r--s',
           linewidth=2, markersize=6, label='ELL 3D: P[70-75]', alpha=0.8)
    
    ax.set_xlabel('Catalog depth zmax', fontsize=12)
    ax.set_ylabel('Probability mass', fontsize=12)
    ax.set_title(f'Probability Mass in CMB/Local Windows vs Depth: {event_id}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    out_path = project_root / "figures" / f"h0_prob_mass_windows_vs_depth_{event_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {out_path}", flush=True)


def plot_cmb_local_ratio(df_metrics: pd.DataFrame, event_id: str, project_root: Path):
    """Plot log10(R) = log10(P_CMB/P_Local) vs zmax."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_metrics = df_metrics.sort_values('zmax')
    
    # Compute log10(R) with regularization
    logR_std = np.log10(np.maximum(df_metrics['R_std'], 1e-12))
    logR_ell = np.log10(np.maximum(df_metrics['R_ell'], 1e-12))
    
    # Plot STD and ELL
    ax.plot(df_metrics['zmax'], logR_std, 'b-o', 
           linewidth=2, markersize=6, label='STD 3D: log₁₀(P[65-70]/P[70-75])', alpha=0.8)
    ax.plot(df_metrics['zmax'], logR_ell, 'r--s',
           linewidth=2, markersize=6, label='ELL 3D: log₁₀(P[65-70]/P[70-75])', alpha=0.8)
    
    # Add horizontal line at zero (R=1)
    ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5, label='R=1 (equal masses)')
    
    ax.set_xlabel('Catalog depth zmax', fontsize=12)
    ax.set_ylabel('log₁₀(R) = log₁₀(P[CMB]/P[Local])', fontsize=12)
    ax.set_title(f'CMB/Local Ratio vs Catalog Depth: {event_id}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    out_path = project_root / "figures" / f"h0_cmb_local_ratio_vs_depth_{event_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {out_path}", flush=True)


def plot_delta_mass(df_metrics: pd.DataFrame, event_id: str, project_root: Path):
    """Plot DeltaP = P_CMB - P_Local vs zmax for STD and ELL."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df_metrics = df_metrics.sort_values('zmax')
    
    # Plot STD and ELL
    ax.plot(df_metrics['zmax'], df_metrics['DeltaP_std'], 'b-o',
           linewidth=2, markersize=6, label='STD 3D: ΔP = P[65-70] - P[70-75]', alpha=0.8)
    ax.plot(df_metrics['zmax'], df_metrics['DeltaP_ell'], 'r--s',
           linewidth=2, markersize=6, label='ELL 3D: ΔP = P[65-70] - P[70-75]', alpha=0.8)
    
    # Add horizontal line at zero
    ax.axhline(0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Catalog depth zmax', fontsize=12)
    ax.set_ylabel('ΔP = P[CMB] - P[Local]', fontsize=12)
    ax.set_title(f'Probability Mass Difference vs Catalog Depth: {event_id}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    out_path = project_root / "figures" / f"h0_delta_mass_vs_depth_{event_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved: {out_path}", flush=True)


def main():
    """Main function for depth scan analysis."""
    parser = argparse.ArgumentParser(
        description="Depth scan: H0 posterior vs catalog depth (zmax) for 3D skymap mode"
    )
    parser.add_argument(
        '--event',
        type=str,
        required=True,
        help='Event name (must exist in manifest)'
    )
    parser.add_argument(
        '--zmax-list',
        type=float,
        nargs='+',
        default=[0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5],
        help='List of zmax values to scan (default: 0.01 0.02 0.03 0.05 0.07 0.1 0.15 0.2 0.3 0.5)'
    )
    parser.add_argument(
        '--h0-min',
        type=float,
        default=20.0,
        help='Minimum H0 value (default: 20)'
    )
    parser.add_argument(
        '--h0-max',
        type=float,
        default=120.0,
        help='Maximum H0 value (default: 120)'
    )
    parser.add_argument(
        '--h0-n',
        type=int,
        default=201,
        help='Number of H0 grid points (default: 201)'
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
        '--out-fig',
        type=str,
        default=None,
        help='Output figure path (default: figures/h0_depth_scan_<event>.png)'
    )
    parser.add_argument(
        '--out-csv',
        type=str,
        default=None,
        help='Output CSV path (default: results/h0_depth_scan_<event>.csv)'
    )
    
    args = parser.parse_args()
    
    # Check availability
    if not ASTROPY_AVAILABLE:
        print("ERROR: astropy and astropy-healpix required")
        print("  Install with: pip install astropy astropy-healpix")
        return
    
    # Paths
    project_root = Path(__file__).parent.parent
    manifest_path = project_root / args.manifest
    galaxy_path = project_root / args.galaxies
    
    # Output paths
    if args.out_fig:
        out_fig_path = Path(args.out_fig)
    else:
        out_fig_path = project_root / "figures" / f"h0_depth_scan_{args.event}.png"
    
    # Results directory
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)
    
    if args.out_csv:
        out_csv_path = Path(args.out_csv)
    else:
        out_csv_path = results_dir / f"h0_depth_scan_{args.event}.csv"
    
    # Model list: STD and ELL with kappa=1.0
    model_list = [("std", 0.0), ("ell", 1.0)]
    
    # Run depth scan
    df_results, H0_grid, posteriors_std_3d, posteriors_ell_3d, zmax_list_valid = run_depth_scan(
        event_id=args.event,
        zmax_list=args.zmax_list,
        model_list=model_list,
        H0_min=args.h0_min,
        H0_max=args.h0_max,
        H0_n=args.h0_n,
        galaxy_path=galaxy_path,
        manifest_path=manifest_path,
        project_root=project_root
    )
    
    # Results directory (already created above)
    results_dir = project_root / "results"
    
    # Save posteriors to NPZ
    if posteriors_std_3d is not None and posteriors_ell_3d is not None:
        npz_path = results_dir / f"h0_depth_scan_{args.event}_posteriors.npz"
        np.savez(
            npz_path,
            zmax_list=np.array(zmax_list_valid),
            h0_grid=H0_grid,
            post_std_3d=posteriors_std_3d,
            post_ell_3d=posteriors_ell_3d
        )
        print(f"\nPosteriors saved to NPZ: {npz_path}", flush=True)
    
    # Compute comparative metrics
    df_std = df_results[df_results['model'] == 'std'].copy().sort_values('zmax')
    df_ell = df_results[df_results['model'] == 'ell'].copy().sort_values('zmax')
    
    # Merge on zmax to compute deltas
    df_metrics = pd.merge(
        df_std[['zmax', 'H0_MAP', 'H0_median', 'H0_p16', 'H0_p84', 
                'P_CMB', 'P_Local', 'R', 'DeltaP', 'n_galaxies']],
        df_ell[['zmax', 'H0_MAP', 'H0_median', 'H0_p16', 'H0_p84',
                'P_CMB', 'P_Local', 'R', 'DeltaP']],
        on='zmax', suffixes=('_std', '_ell')
    )
    
    # Compute comparative metrics
    df_metrics['dP_CMB'] = df_metrics['P_CMB_ell'] - df_metrics['P_CMB_std']
    df_metrics['dP_Local'] = df_metrics['P_Local_ell'] - df_metrics['P_Local_std']
    df_metrics['dR'] = df_metrics['R_ell'] - df_metrics['R_std']
    df_metrics['dMedian'] = df_metrics['H0_median_ell'] - df_metrics['H0_median_std']
    df_metrics['dMAP'] = df_metrics['H0_MAP_ell'] - df_metrics['H0_MAP_std']
    
    # Save metrics CSV
    metrics_csv_path = results_dir / f"h0_depth_scan_{args.event}_metrics.csv"
    df_metrics.to_csv(metrics_csv_path, index=False)
    print(f"Metrics saved to CSV: {metrics_csv_path}", flush=True)
    
    # Save original CSV
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(out_csv_path, index=False)
    print(f"Results saved to CSV: {out_csv_path}", flush=True)
    
    # Plot original results
    plot_depth_scan(df_results, args.event, out_fig_path)
    
    # Plot new metric figures
    plot_prob_mass_windows(df_metrics, args.event, project_root)
    plot_cmb_local_ratio(df_metrics, args.event, project_root)
    plot_delta_mass(df_metrics, args.event, project_root)
    
    # Print summary
    print(f"\n{'='*60}", flush=True)
    print(f"Summary", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total zmax values scanned: {len(args.zmax_list)}", flush=True)
    print(f"Total results: {len(df_results)}", flush=True)
    
    # Print metrics summary
    if len(df_metrics) > 0:
        print(f"\nMetrics Summary:", flush=True)
        print(f"{'='*60}", flush=True)
        
        # Find zmax where ELL increases P_CMB relative to STD
        df_metrics['P_CMB_increase'] = df_metrics['dP_CMB'] > 0
        zmax_increase = df_metrics[df_metrics['P_CMB_increase']]['zmax'].tolist()
        if len(zmax_increase) > 0:
            print(f"zmax where ELL increases P[CMB] relative to STD: {zmax_increase}", flush=True)
        else:
            print(f"ELL never increases P[CMB] relative to STD", flush=True)
        
        # Find maximum |dP_CMB|
        max_dP_CMB_idx = df_metrics['dP_CMB'].abs().idxmax()
        max_dP_CMB = df_metrics.loc[max_dP_CMB_idx, 'dP_CMB']
        zmax_max_dP = df_metrics.loc[max_dP_CMB_idx, 'zmax']
        print(f"Maximum |dP_CMB| = {max_dP_CMB:.4f} at zmax = {zmax_max_dP:.3f}", flush=True)
        
        # Find maximum |dR| (in log10 space)
        logR_std = np.log10(np.maximum(df_metrics['R_std'], 1e-12))
        logR_ell = np.log10(np.maximum(df_metrics['R_ell'], 1e-12))
        dlogR = logR_ell - logR_std
        max_dlogR_idx = dlogR.abs().idxmax()
        max_dlogR = dlogR.loc[max_dlogR_idx]
        zmax_max_dlogR = df_metrics.loc[max_dlogR_idx, 'zmax']
        print(f"Maximum |dlog₁₀(R)| = {max_dlogR:.4f} at zmax = {zmax_max_dlogR:.3f}", flush=True)
        
        print(f"\nMetrics table preview:", flush=True)
        print(df_metrics[['zmax', 'P_CMB_std', 'P_CMB_ell', 'dP_CMB', 
                          'R_std', 'R_ell', 'dR', 'dMedian']].to_string(index=False), flush=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"SUCCESS: Depth scan completed", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{'='*60}", flush=True)
        print(f"INTERRUPTED by user", flush=True)
        print(f"{'='*60}", flush=True)
        sys.exit(130)
    except Exception as e:
        print(f"\n{'='*60}", flush=True)
        print(f"FATAL ERROR: {e}", flush=True)
        print(f"{'='*60}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

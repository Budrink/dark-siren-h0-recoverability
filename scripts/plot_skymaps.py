"""Plot skymaps from manifest.json."""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Try to import ligo.skymap (preferred)
try:
    from ligo.skymap.io.fits import read_sky_map
    from ligo.skymap.postprocess import find_greedy_credible_levels
    LIGO_SKYMAP_AVAILABLE = True
except ImportError:
    LIGO_SKYMAP_AVAILABLE = False

# Fallback: astropy and astropy-healpix
try:
    from astropy.io import fits
    from astropy_healpix import HEALPix
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False


def load_manifest(manifest_path: Path) -> list:
    """Load manifest from JSON file."""
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return []
    
    with open(manifest_path, 'r') as f:
        return json.load(f)


def save_manifest(manifest: list, manifest_path: Path) -> None:
    """Save manifest to JSON file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def read_skymap_ligo(fits_path: Path) -> Tuple[np.ndarray, dict]:
    """Read skymap using ligo.skymap."""
    prob, metadata = read_sky_map(str(fits_path))
    return prob, metadata


def read_skymap_astropy(fits_path: Path, target_nside: int = 512) -> Tuple[np.ndarray, dict]:
    """Read skymap using astropy and convert UNIQ to regular HEALPix."""
    with fits.open(str(fits_path)) as hdul:
        hdu = hdul[1]
        prob = hdu.data['PROBDENSITY']
        uniq = hdu.data['UNIQ']
        
        # Convert UNIQ to regular HEALPix grid
        hp_target = HEALPix(nside=target_nside, order='nested', frame='icrs')
        npix_target = hp_target.npix
        prob_map = np.zeros(npix_target)
        
        # Decode UNIQ values and accumulate probabilities
        for i in range(len(uniq)):
            u = uniq[i]
            p = prob[i] if i < len(prob) else 0.0
            
            # Decode UNIQ: find nside such that 4*nside^2 <= u < 4*(2*nside)^2
            level = 0
            while 4 * (4 ** (level + 1)) <= u:
                level += 1
            
            nside_u = 2 ** level
            ipix_u = u - 4 * (4 ** level)
            
            # Get pixel center coordinates
            hp_source = HEALPix(nside=nside_u, order='nested', frame='icrs')
            lon_u, lat_u = hp_source.healpix_to_lonlat([ipix_u])
            
            # Find target pixel containing this coordinate
            ipix_target = hp_target.lonlat_to_healpix(lon_u, lat_u)[0]
            
            # Accumulate probability (use max to handle overlaps)
            if ipix_target < npix_target:
                prob_map[ipix_target] = max(prob_map[ipix_target], p)
        
        # Normalize
        if prob_map.sum() > 0:
            prob_map = prob_map / prob_map.sum() * len(prob) * np.mean(prob) if len(prob) > 0 else prob_map
        
        metadata = {'nside': target_nside, 'nest': True}
        return prob_map, metadata


def compute_sky_area_90(prob: np.ndarray, metadata: dict) -> float:
    """
    Compute 90% credible sky area in deg^2.
    
    Parameters
    ----------
    prob : np.ndarray
        Probability map (probability density, not normalized probabilities)
    metadata : dict
        Metadata with nside and nest info
    
    Returns
    -------
    float
        Sky area in deg^2
    """
    # Get nside from metadata
    nside = metadata.get('nside', 512)
    npix_total = 12 * nside * nside
    
    # Area per pixel in steradians
    area_per_pixel = 4 * np.pi / npix_total
    
    # Convert probability density to probability per pixel
    # prob is probability density, so multiply by pixel area to get probability
    prob_per_pixel = prob * area_per_pixel
    
    # Normalize to ensure sum = 1
    total_prob = prob_per_pixel.sum()
    if total_prob > 0:
        prob_per_pixel = prob_per_pixel / total_prob
    else:
        return 0.0
    
    # Sort probabilities in descending order
    prob_sorted = np.sort(prob_per_pixel)[::-1]
    
    # Cumulative sum
    cumsum = np.cumsum(prob_sorted)
    
    # Find pixels that contain 90% of probability
    threshold_idx = np.searchsorted(cumsum, 0.9)
    
    # Number of pixels in 90% credible region
    npix_90 = threshold_idx + 1
    
    # Total area in steradians
    area_sr = npix_90 * area_per_pixel
    
    # Convert to deg^2 (1 sr = (180/pi)^2 deg^2)
    area_deg2 = area_sr * (180.0 / np.pi) ** 2
    
    return area_deg2


def plot_skymap_ligo(prob: np.ndarray, metadata: dict, fig_path: Path, 
                    title: str, cmap: str = 'plasma') -> bool:
    """Plot skymap using ligo.skymap."""
    try:
        nested = metadata.get('nest', False)
        
        # Create figure
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection='astro hours mollweide')
        ax.grid(color='gray', ls='--', lw=0.5, alpha=0.5)
        
        # Display the HEALPix probability map
        img = ax.imshow_hpx((prob, 'ICRS'), nested=nested,
                            cmap=cmap, norm=None)
        
        # Add colorbar
        cb = fig.colorbar(img, ax=ax, orientation='horizontal', 
                         pad=0.05, shrink=0.8)
        cb.set_label('Probability Density')
        
        # Set title
        ax.set_title(title, pad=20)
        
        plt.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"  ERROR: Failed to plot with ligo.skymap: {e}")
        return False


def plot_skymap_astropy(prob: np.ndarray, metadata: dict, fig_path: Path,
                       title: str, cmap: str = 'plasma') -> bool:
    """Plot skymap using astropy and matplotlib."""
    try:
        nside = metadata.get('nside', 512)
        nested = metadata.get('nest', False)
        
        # Create HEALPix object
        hp = HEALPix(nside=nside, order='nested' if nested else 'ring', frame='icrs')
        
        # Get coordinates for all pixels
        npix = len(prob)
        indices = np.arange(npix)
        lon, lat = hp.healpix_to_lonlat(indices)
        
        # Create figure with Mollweide projection
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='mollweide')
        
        # Convert to radians for matplotlib
        lon_rad = np.radians(lon.value)
        lat_rad = np.radians(lat.value)
        
        # Create scatter plot with color mapping
        scatter = ax.scatter(lon_rad, lat_rad, c=prob, 
                           cmap=cmap, s=1, alpha=0.8, 
                           vmin=0, vmax=np.max(prob) if len(prob) > 0 else 1)
        
        # Add colorbar
        cb = fig.colorbar(scatter, ax=ax, orientation='horizontal', 
                         pad=0.05, shrink=0.8)
        cb.set_label('Probability Density')
        
        # Set labels
        ax.set_xlabel('RA', labelpad=20)
        ax.set_ylabel('Dec', labelpad=20)
        ax.set_title(title, pad=20)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        return True
    except Exception as e:
        print(f"  ERROR: Failed to plot with astropy: {e}")
        import traceback
        traceback.print_exc()
        return False


def extract_distance_pdf(fits_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract distance PDF from 3D FITS skymap with proper marginalization.
    
    For 3D skymaps with DISTMU, DISTSIGMA, DISTNORM, PROBDENSITY columns:
    - Computes conditional density p(r|pixel) = DISTNORM * r^2 * exp(-(r-mu)^2/(2*sigma^2))
    - Marginalizes over sky: p(r) = sum_pix PROB[pix] * p(r|pix)
    - Normalizes to unit integral
    
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (r_grid, p_r_post) - distance grid (Mpc) and normalized posterior PDF
    """
    # Try with astropy (fallback if ligo.skymap not available)
    with fits.open(str(fits_path)) as hdul:
        hdu = hdul[1]
        
        # Check for 3D distance information
        if 'DISTMU' not in hdu.data.dtype.names:
            raise ValueError("Distance information (DISTMU) not found in FITS file")
        
        # Extract arrays
        # Note: column is PROBDENSITY (probability density), not PROB
        prob = hdu.data['PROBDENSITY']  # Sky probability density for each pixel
        mu = hdu.data['DISTMU']  # Distance mean (Mpc, linear space)
        sigma = hdu.data['DISTSIGMA']  # Distance sigma (Mpc, linear space)
        norm = hdu.data['DISTNORM']  # Normalization constant (Mpc^-2)
        
        # According to LIGO skymap format:
        # p(r|pix) = DISTNORM * r^2 * exp(-(r - DISTMU)^2 / (2*DISTSIGMA^2))
        # where DISTMU and DISTSIGMA are in linear Mpc space
        
        # Filter valid pixels
        valid_mask = (
            np.isfinite(prob) & np.isfinite(mu) & np.isfinite(sigma) & np.isfinite(norm) &
            (sigma > 0) & (prob > 0) & (mu > 0)
        )
        
        if not np.any(valid_mask):
            raise ValueError("No valid distance data found in FITS")
        
        prob_valid = prob[valid_mask]
        mu_valid = mu[valid_mask]
        sigma_valid = sigma[valid_mask]
        norm_valid = norm[valid_mask]
        
        # Normalize sky probabilities (they should already be normalized, but ensure it)
        prob_valid = prob_valid / np.sum(prob_valid)
        
        # Build distance grid (r_grid in Mpc)
        # Start from r_min = 1.0 Mpc (avoid r=0 for prior removal)
        # r_max: use 2-3 * median distance or max reasonable value
        mu_median = np.median(mu_valid)  # Already in linear Mpc
        sigma_max = np.max(sigma_valid)
        r_min = 1.0  # Mpc
        r_max = max(3.0 * mu_median, np.max(mu_valid) + 5 * sigma_max)
        r_max = min(r_max, 50000.0)  # Cap at 50 Gpc (reasonable upper limit)
        
        N_grid = 1000
        r_grid = np.linspace(r_min, r_max, N_grid)
        
        # Compute posterior: p(r) = sum_pix PROBDENSITY[pix] * p(r|pix)
        # where p(r|pix) = DISTNORM * r^2 * exp(-(r - DISTMU)^2 / (2*DISTSIGMA^2))
        p_r_post = np.zeros_like(r_grid)
        
        # Vectorized computation over pixels
        for i in range(len(prob_valid)):
            # Conditional density for this pixel
            # p(r|pix) = norm * r^2 * exp(-(r - mu)^2 / (2*sigma^2))
            diff = (r_grid - mu_valid[i]) / sigma_valid[i]
            p_r_given_pix = norm_valid[i] * (r_grid ** 2) * np.exp(-0.5 * diff ** 2)
            
            # Marginalize: add weighted contribution
            p_r_post += prob_valid[i] * p_r_given_pix
        
        # Normalize posterior
        norm_post = np.trapezoid(p_r_post, r_grid)
        if norm_post > 0:
            p_r_post = p_r_post / norm_post
        else:
            raise ValueError("Distance PDF normalization failed")
        
        return r_grid, p_r_post


def plot_distance_posterior(fits_path: Path, fig_path: Path, 
                           title: str, cmap: str = 'plasma',
                           save_pdf: bool = False, pdf_path: Path = None,
                           event_name: str = None) -> bool:
    """
    Plot distance posterior if available in FITS.
    
    Parameters
    ----------
    fits_path : Path
        Path to FITS file
    fig_path : Path
        Output figure path
    title : str
        Plot title
    cmap : str
        Colormap (not used for distance plot)
    save_pdf : bool
        Whether to save PDF to NPZ file
    pdf_path : Path
        Path to save PDF NPZ file
    
    Returns
    -------
    bool
        True if successful
    """
    try:
        d_grid, pdf_grid = extract_distance_pdf(fits_path)
        
        # Create histogram
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(d_grid, pdf_grid, 'b-', linewidth=2, alpha=0.7)
        ax.fill_between(d_grid, pdf_grid, alpha=0.3, color='steelblue')
        ax.set_xlabel('Distance (Mpc)')
        ax.set_ylabel('Probability Density')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        # Save PDF if requested
        if save_pdf and pdf_path is not None:
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure normalization of posterior
            p_post = pdf_grid.copy()
            d_grid = d_grid.copy()  # Rename for clarity (it's r_grid)
            
            norm_post = np.trapezoid(p_post, d_grid)
            if abs(norm_post - 1.0) > 0.01:
                print(f"  WARNING: Posterior normalization = {norm_post:.6f}, renormalizing...")
                p_post = p_post / norm_post
            
            # Compute GW likelihood by removing d^2 prior
            # Prior: π(d) ∝ d^2 (uniform in comoving volume)
            # Likelihood: p_like(d) = p_post(d) / d^2 (then renormalize)
            # Note: d_grid starts from r_min = 1.0 Mpc, so no division by zero
            
            p_like = p_post / (d_grid ** 2)
            
            # Normalize likelihood
            norm_like = np.trapezoid(p_like, d_grid)
            if norm_like > 0:
                p_like = p_like / norm_like
            else:
                print(f"  WARNING: Likelihood normalization failed, using posterior")
                p_like = p_post.copy()
            
            # Compute statistics for posterior
            dx = d_grid[1] - d_grid[0] if len(d_grid) > 1 else 1.0
            
            # Mode
            d_mode_post_idx = np.argmax(p_post)
            d_mode_post = d_grid[d_mode_post_idx]
            
            # Median and quantiles
            cdf_post = np.cumsum(p_post) * dx
            median_post_idx = np.searchsorted(cdf_post, 0.5)
            median_post_idx = min(median_post_idx, len(d_grid) - 1)
            d_median_post = d_grid[median_post_idx]
            
            q5_post_idx = np.searchsorted(cdf_post, 0.05)
            q5_post_idx = min(q5_post_idx, len(d_grid) - 1)
            d_q5_post = d_grid[q5_post_idx]
            
            q50_post_idx = np.searchsorted(cdf_post, 0.50)
            q50_post_idx = min(q50_post_idx, len(d_grid) - 1)
            d_q50_post = d_grid[q50_post_idx]
            
            q95_post_idx = np.searchsorted(cdf_post, 0.95)
            q95_post_idx = min(q95_post_idx, len(d_grid) - 1)
            d_q95_post = d_grid[q95_post_idx]
            
            # Compute statistics for likelihood
            d_mode_like_idx = np.argmax(p_like)
            d_mode_like = d_grid[d_mode_like_idx]
            
            cdf_like = np.cumsum(p_like) * dx
            median_like_idx = np.searchsorted(cdf_like, 0.5)
            median_like_idx = min(median_like_idx, len(d_grid) - 1)
            d_median_like = d_grid[median_like_idx]
            
            q5_like_idx = np.searchsorted(cdf_like, 0.05)
            q5_like_idx = min(q5_like_idx, len(d_grid) - 1)
            d_q5_like = d_grid[q5_like_idx]
            
            q50_like_idx = np.searchsorted(cdf_like, 0.50)
            q50_like_idx = min(q50_like_idx, len(d_grid) - 1)
            d_q50_like = d_grid[q50_like_idx]
            
            q95_like_idx = np.searchsorted(cdf_like, 0.95)
            q95_like_idx = min(q95_like_idx, len(d_grid) - 1)
            d_q95_like = d_grid[q95_like_idx]
            
            # Print diagnostics
            print(f"  Distance PDF diagnostics:")
            print(f"    Posterior: mode={d_mode_post:.1f} Mpc, median={d_median_post:.1f} Mpc")
            print(f"      Quantiles: 5%={d_q5_post:.1f}, 50%={d_q50_post:.1f}, 95%={d_q95_post:.1f} Mpc")
            print(f"    Likelihood (d^2 prior removed): mode={d_mode_like:.1f} Mpc, median={d_median_like:.1f} Mpc")
            print(f"      Quantiles: 5%={d_q5_like:.1f}, 50%={d_q50_like:.1f}, 95%={d_q95_like:.1f} Mpc")
            
            # Save NPZ with both posterior and likelihood
            np.savez(pdf_path, d_grid=d_grid, p_post=p_post, p_like=p_like)
            
            # Save metadata JSON (sidecar file)
            import json
            meta_path = pdf_path.parent / "distance_pdf_meta.json"
            
            metadata = {
                "event": event_name if event_name else "unknown",
                "distance_unit": "Mpc",
                "post_norm": float(np.trapezoid(p_post, d_grid)),
                "like_norm": float(np.trapezoid(p_like, d_grid)),
                "d_min": float(d_grid[0]),
                "d_max": float(d_grid[-1]),
                "d_mode_post": float(d_mode_post),
                "d_median_post": float(d_median_post),
                "d_q5_post": float(d_q5_post),
                "d_q50_post": float(d_q50_post),
                "d_q95_post": float(d_q95_post),
                "d_mode_like": float(d_mode_like),
                "d_median_like": float(d_median_like),
                "d_q5_like": float(d_q5_like),
                "d_q50_like": float(d_q50_like),
                "d_q95_like": float(d_q95_like),
                "prior_removed": "d^2",
                "r_min": float(d_grid[0])
            }
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  Saved distance PDF to: {pdf_path}")
            
            # Create debug plot with both distributions
            debug_fig_path = pdf_path.parent / f"distance_pdf_debug_{event_name if event_name else 'unknown'}.png"
            fig_debug, ax_debug = plt.subplots(figsize=(10, 6))
            
            ax_debug.plot(d_grid, p_post, 'b-', linewidth=2, label='Posterior (with d^2 prior)', alpha=0.8)
            ax_debug.fill_between(d_grid, p_post, alpha=0.3, color='blue')
            
            ax_debug.plot(d_grid, p_like, 'r--', linewidth=2, label='Likelihood (prior removed)', alpha=0.8)
            ax_debug.fill_between(d_grid, p_like, alpha=0.2, color='red')
            
            # Add vertical lines at mode and median
            ax_debug.axvline(d_mode_post, color='blue', linestyle=':', linewidth=1, alpha=0.7, label=f'Posterior mode={d_mode_post:.1f} Mpc')
            ax_debug.axvline(d_median_post, color='blue', linestyle='-.', linewidth=1, alpha=0.7, label=f'Posterior median={d_median_post:.1f} Mpc')
            ax_debug.axvline(d_mode_like, color='red', linestyle=':', linewidth=1, alpha=0.7, label=f'Likelihood mode={d_mode_like:.1f} Mpc')
            ax_debug.axvline(d_median_like, color='red', linestyle='-.', linewidth=1, alpha=0.7, label=f'Likelihood median={d_median_like:.1f} Mpc')
            
            ax_debug.set_xlabel('Distance (Mpc)')
            ax_debug.set_ylabel('Probability Density')
            ax_debug.set_title(f'Distance PDF: {event_name if event_name else "unknown"}')
            ax_debug.legend(loc='best', fontsize=9)
            ax_debug.grid(True, alpha=0.3)
            ax_debug.set_xlim(d_grid[0], min(d_grid[-1], d_q95_post * 1.5))  # Focus on relevant range
            
            plt.tight_layout()
            plt.savefig(debug_fig_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            print(f"  Debug plot saved: {debug_fig_path}")
            print(f"  Saved metadata to: {meta_path}")
        
        return True
    except Exception as e:
        print(f"  ERROR: Failed to plot distance posterior: {e}")
        return False


def main():
    """Main function to plot skymaps from manifest."""
    parser = argparse.ArgumentParser(
        description="Plot skymaps from manifest.json"
    )
    parser.add_argument(
        '--manifest',
        type=str,
        default='data/skymaps/manifest.json',
        help='Path to manifest JSON file (default: data/skymaps/manifest.json)'
    )
    parser.add_argument(
        '--cmap',
        type=str,
        default=None,
        help='Colormap name (default: use matplotlib default or plasma)'
    )
    parser.add_argument(
        '--distance',
        action='store_true',
        help='Also plot distance posterior if available'
    )
    parser.add_argument(
        '--area',
        action='store_true',
        help='Compute and print sky_area_90 from probability map'
    )
    args = parser.parse_args()
    
    # Determine colormap
    cmap = args.cmap if args.cmap else 'plasma'
    
    # Check availability
    if not LIGO_SKYMAP_AVAILABLE and not ASTROPY_AVAILABLE:
        print("ERROR: No plotting libraries available.")
        print("  Install with: pip install astropy astropy-healpix")
        print("  Or for better plotting: pip install ligo-skymap astropy astropy-healpix")
        return
    
    # Paths
    project_root = Path(__file__).parent.parent
    manifest_path = project_root / args.manifest
    figure_dir = project_root / "figures" / "skymaps"
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest
    manifest = load_manifest(manifest_path)
    if len(manifest) == 0:
        print("ERROR: No events in manifest")
        return
    
    print(f"Loaded {len(manifest)} events from manifest")
    
    # Process each event
    for entry in manifest:
        event_name = entry.get('event')
        # Support both 'fits_path' (new) and 'extracted_fits_path' (old) for compatibility
        fits_path_rel = entry.get('fits_path') or entry.get('extracted_fits_path')
        
        if not event_name or not fits_path_rel:
            print(f"  Skipping entry with missing event or fits_path")
            continue
        
        fits_path = project_root / fits_path_rel
        
        if not fits_path.exists():
            print(f"  Skipping {event_name}: FITS file not found: {fits_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing event: {event_name}")
        print(f"{'='*60}")
        
        # Read skymap
        print(f"Step 1: Read skymap")
        try:
            if LIGO_SKYMAP_AVAILABLE:
                prob, metadata = read_skymap_ligo(fits_path)
                print(f"  Using ligo.skymap")
            else:
                prob, metadata = read_skymap_astropy(fits_path)
                print(f"  Using astropy fallback")
        except Exception as e:
            print(f"  ERROR: Failed to read skymap: {e}")
            continue
        
        # Plot skymap
        print(f"Step 2: Plot skymap")
        figure_skymap_path = figure_dir / f"skymap_{event_name}.png"
        title = f"{event_name} sky probability (2D)"
        
        plot_success = False
        if LIGO_SKYMAP_AVAILABLE:
            plot_success = plot_skymap_ligo(prob, metadata, figure_skymap_path, title, cmap)
        else:
            plot_success = plot_skymap_astropy(prob, metadata, figure_skymap_path, title, cmap)
        
        if plot_success:
            print(f"  Plot saved: {figure_skymap_path}")
            entry['figure_skymap_path'] = str(figure_skymap_path.relative_to(project_root))
        else:
            print(f"  Plot failed")
            entry['figure_skymap_path'] = None
        
        # Compute sky area if requested
        if args.area:
            print(f"Step 3: Compute sky_area_90")
            try:
                area_90 = compute_sky_area_90(prob, metadata)
                print(f"  sky_area_90 = {area_90:.1f} deg²")
                entry['sky_area_90_from_map'] = area_90
            except Exception as e:
                print(f"  ERROR: Failed to compute sky area: {e}")
                entry['sky_area_90_from_map'] = None
        
        # Plot distance if requested
        if args.distance:
            print(f"Step 4: Plot distance posterior")
            figure_distance_path = figure_dir / f"distance_{event_name}.png"
            title_dist = f"{event_name} distance posterior"
            
            # Save PDF to event directory
            event_dir = fits_path.parent
            pdf_path = event_dir / "distance_pdf.npz"
            
            if plot_distance_posterior(fits_path, figure_distance_path, title_dist, cmap,
                                    save_pdf=True, pdf_path=pdf_path, event_name=event_name):
                print(f"  Plot saved: {figure_distance_path}")
                entry['figure_distance_path'] = str(figure_distance_path.relative_to(project_root))
                entry['distance_pdf_path'] = str(pdf_path.relative_to(project_root))
            else:
                print(f"  Distance plot not available or failed")
                entry['figure_distance_path'] = None
                entry['distance_pdf_path'] = None
    
    # Save updated manifest
    print(f"\n{'='*60}")
    print(f"Saving updated manifest to: {manifest_path}")
    save_manifest(manifest, manifest_path)
    
    print(f"\nDone! Processed {len(manifest)} events.")


if __name__ == "__main__":
    main()

"""Download and plot skymaps for top-N localized GW events."""

import sys
import argparse
import json
import tarfile
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# Try to import astropy and astropy-healpix for plotting
try:
    from astropy.io import fits
    from astropy_healpix import HEALPix
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    SKYMAP_AVAILABLE = True
except ImportError as e:
    SKYMAP_AVAILABLE = False
    print("Warning: Required packages not available. Plotting will be skipped.")
    print(f"  Missing: {e}")
    print("  Install with: pip install astropy astropy-healpix")


def select_events(df: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """
    Select top-N events with available skymaps, ranked by smallest sky_area_90.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with event data
    top_n : int, optional
        Number of top events to select, by default 3
    
    Returns
    -------
    pd.DataFrame
        Selected events DataFrame
    """
    # Filter to events with skymaps
    df_available = df[
        (df['skymap_available'] == True) & 
        (df['skymap_url'].notna()) & 
        (df['skymap_url'] != '')
    ].copy()
    
    if len(df_available) == 0:
        return pd.DataFrame()
    
    # Sort by sky_area_90 ascending (NaNs last)
    df_available['_sort_sky'] = df_available['sky_area_90'].fillna(np.inf)
    df_sorted = df_available.sort_values('_sort_sky', ascending=True)
    df_sorted = df_sorted.drop(columns=['_sort_sky'])
    
    # Take top N
    return df_sorted.head(top_n)


def download_tarball(url: str, path: Path, force: bool = False, 
                    timeout: int = 120, chunk_size: int = 1048576) -> bool:
    """
    Download a tarball from URL with streaming.
    
    Parameters
    ----------
    url : str
        URL to download
    path : Path
        Output file path
    force : bool, optional
        Re-download even if file exists, by default False
    timeout : int, optional
        Request timeout in seconds, by default 120
    chunk_size : int, optional
        Chunk size for streaming in bytes (1MB), by default 1048576
    
    Returns
    -------
    bool
        True if download successful, False otherwise
    """
    # Check if file exists
    if path.exists() and not force:
        print(f"  Using cached tarball: {path}")
        return True
    
    # Create parent directory
    path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"  Downloading from {url}...")
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"  Progress: {percent:.1f}%", end='\r')
        
        print(f"  Downloaded: {path} ({downloaded / 1024 / 1024:.1f} MB)")
        return True
    
    except Exception as e:
        print(f"  ERROR: Failed to download: {e}")
        if path.exists():
            path.unlink()  # Remove partial download
        return False


def find_skymap_member(tar: tarfile.TarFile, event_name: str) -> Optional[tarfile.TarInfo]:
    """
    Find the best skymap FITS file member in a tar archive for a specific event.
    
    Heuristics:
    - Prefer filenames containing the event name (cleaned, without version suffix)
    - Prefer filenames containing "skymap" or "bayestar" (case-insensitive)
    - Prefer .fits.gz over .fits
    - If multiple candidates, pick the smallest path-depth or first match
    - Ignore non-FITS files
    
    Parameters
    ----------
    tar : tarfile.TarFile
        Open tarfile object
    event_name : str
        Event name to match (e.g., "GW191204_171526-v1")
    
    Returns
    -------
    Optional[tarfile.TarInfo]
        Best skymap member or None if not found
    """
    # Clean event name (remove version suffix)
    event_name_clean = event_name.split('-v')[0] if '-v' in event_name else event_name
    event_name_clean_lower = event_name_clean.lower()
    
    candidates = []
    
    for member in tar.getmembers():
        if not member.isfile():
            continue
        
        name_lower = member.name.lower()
        
        # Must be a FITS file
        if not (name_lower.endswith('.fits') or name_lower.endswith('.fits.gz')):
            continue
        
        # Check if filename contains event name
        has_event_name = event_name_clean_lower in name_lower
        
        # Check for skymap/bayestar keywords
        has_keyword = ('skymap' in name_lower or 'bayestar' in name_lower)
        
        # Prefer .fits.gz over .fits
        is_gz = name_lower.endswith('.fits.gz')
        
        # Count path depth (number of slashes)
        depth = member.name.count('/')
        
        candidates.append({
            'member': member,
            'has_event_name': has_event_name,
            'has_keyword': has_keyword,
            'is_gz': is_gz,
            'depth': depth,
            'name': member.name
        })
    
    if not candidates:
        return None
    
    # Sort: event name match first, then keyword match, then .fits.gz, then depth, then name
    candidates.sort(key=lambda x: (
        not x['has_event_name'],  # False first (has event name)
        not x['has_keyword'],     # False first (has keyword)
        not x['is_gz'],           # False first (is gz)
        x['depth'],               # Lower depth first
        x['name']                 # Alphabetical
    ))
    
    return candidates[0]['member']


def extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, 
                  out_path: Path, force: bool = False) -> bool:
    """
    Extract a member from tar archive to output path.
    
    Parameters
    ----------
    tar : tarfile.TarFile
        Open tarfile object
    member : tarfile.TarInfo
        Member to extract
    out_path : Path
        Output file path
    force : bool, optional
        Re-extract even if file exists, by default False
    
    Returns
    -------
    bool
        True if extraction successful, False otherwise
    """
    # Check if file exists
    if out_path.exists() and not force:
        print(f"  Using cached FITS: {out_path}")
        return True
    
    # Create parent directory
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        print(f"  Extracting {member.name} to {out_path}...")
        
        # Extract member
        extracted_file = tar.extractfile(member)
        if extracted_file is None:
            print(f"  ERROR: Could not extract member {member.name}")
            return False
        
        # Write to output path
        try:
            with open(out_path, 'wb') as f:
                f.write(extracted_file.read())
        finally:
            extracted_file.close()
        
        print(f"  Extracted: {out_path}")
        return True
    
    except Exception as e:
        print(f"  ERROR: Failed to extract: {e}")
        if out_path.exists():
            out_path.unlink()
        return False


def plot_skymap(fits_path: Path, fig_path: Path, title: str) -> bool:
    """
    Plot a HEALPix skymap using astropy and matplotlib.
    
    Parameters
    ----------
    fits_path : Path
        Path to HEALPix FITS file
    fig_path : Path
        Output figure path
    title : str
        Plot title
    
    Returns
    -------
    bool
        True if plotting successful, False otherwise
    """
    if not SKYMAP_AVAILABLE:
        print(f"  Skipping plot: required packages not available")
        print(f"  Install with: pip install astropy astropy-healpix")
        return False
    
    try:
        # Read FITS file
        with fits.open(str(fits_path)) as hdul:
            hdu = hdul[1]
            prob = hdu.data['PROBDENSITY']
            uniq = hdu.data['UNIQ']
            
            # Decode UNIQ to get nside and ipix for each pixel
            # UNIQ = 4 * nside^2 + ipix, where nside is power of 2
            # We'll convert to a fixed nside grid for visualization
            target_nside = 128  # Reasonable resolution for visualization
            hp_target = HEALPix(nside=target_nside, order='nested', frame='icrs')
            npix_target = hp_target.npix
            prob_map = np.zeros(npix_target)
            
            # Decode UNIQ values and accumulate probabilities
            for i in range(len(uniq)):
                u = uniq[i]
                p = prob[i] if i < len(prob) else 0.0
                
                # Decode UNIQ: find nside such that 4*nside^2 <= u < 4*(2*nside)^2
                # Simplified: extract nside from UNIQ
                # UNIQ format: bits encode nside level and pixel index
                # Level k: nside = 2^k, range is [4*4^k, 4*4^(k+1))
                level = 0
                while 4 * (4 ** (level + 1)) <= u:
                    level += 1
                
                nside_u = 2 ** level
                ipix_u = u - 4 * (4 ** level)
                
                # Convert to target resolution
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
            
            # Get coordinates for target grid
            indices = np.arange(npix_target)
            lon, lat = hp_target.healpix_to_lonlat(indices)
            prob_plot = prob_map
        
        # Create figure with Mollweide projection
        fig = plt.figure(figsize=(12, 6))
        ax = fig.add_subplot(111, projection='mollweide')
        
        # Convert to radians for matplotlib
        lon_rad = np.radians(lon.value)
        lat_rad = np.radians(lat.value)
        
        # Create scatter plot with color mapping
        scatter = ax.scatter(lon_rad, lat_rad, c=prob_plot, 
                           cmap='plasma', s=1, alpha=0.8, 
                           vmin=0, vmax=np.max(prob_plot) if len(prob_plot) > 0 else 1)
        
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
        
        # Create output directory if needed
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  Plot saved: {fig_path}")
        return True
    
    except Exception as e:
        print(f"  ERROR: Failed to plot: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function to download and plot skymaps."""
    parser = argparse.ArgumentParser(
        description="Download and plot skymaps for top-N localized GW events"
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=3,
        help='Number of top events to process (default: 3)'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='data/events_gwtc3_confident.csv',
        help='Path to events CSV file (default: data/events_gwtc3_confident.csv)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Re-download and re-extract even if cached'
    )
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent
    csv_path = project_root / args.csv
    tarball_dir = project_root / "data" / "zenodo_tarballs"
    skymap_dir = project_root / "data" / "skymaps"
    figure_dir = project_root / "figures" / "skymaps"
    manifest_path = skymap_dir / "manifest.json"
    
    # Create directories
    tarball_dir.mkdir(parents=True, exist_ok=True)
    skymap_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV
    print(f"Reading events from: {csv_path}")
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} events from CSV")
    
    # Select top-N events
    print(f"\nSelecting top {args.top_n} events with skymaps...")
    df_selected = select_events(df, top_n=args.top_n)
    
    if len(df_selected) == 0:
        print("ERROR: No events with available skymaps found")
        return
    
    print(f"Selected {len(df_selected)} events:")
    for idx, row in df_selected.iterrows():
        print(f"  {row['event']}: sky_area_90={row['sky_area_90']:.1f} deg²")
    
    # Process each event
    manifest = []
    
    for idx, row in df_selected.iterrows():
        event_name = row['event']
        skymap_url = row['skymap_url']
        sky_area_90 = row['sky_area_90']
        
        print(f"\n{'='*60}")
        print(f"Processing event: {event_name}")
        print(f"{'='*60}")
        
        # Paths for this event
        tarball_path = tarball_dir / f"{event_name}.tar.gz"
        event_skymap_dir = skymap_dir / event_name
        figure_path = figure_dir / f"skymap_{event_name}.png"
        
        # Download tarball
        print(f"Step 1: Download tarball")
        if not download_tarball(skymap_url, tarball_path, force=args.force):
            print(f"  Skipping event {event_name} due to download failure")
            continue
        
        # Extract skymap FITS
        print(f"Step 2: Extract skymap FITS")
        extracted_fits_path = None
        try:
            with tarfile.open(tarball_path, 'r:gz') as tar:
                # Find best skymap member for this event
                skymap_member = find_skymap_member(tar, event_name)
                
                if skymap_member is None:
                    # List top 10 members for debugging
                    members = [m.name for m in tar.getmembers() if m.isfile()][:10]
                    print(f"  ERROR: No FITS skymap found in tarball")
                    print(f"  Top 10 members in archive:")
                    for m in members:
                        print(f"    - {m}")
                    continue
                
                print(f"  Found skymap member: {skymap_member.name}")
                
                # Determine output filename based on member extension
                if skymap_member.name.lower().endswith('.fits.gz'):
                    extracted_fits_path = event_skymap_dir / "skymap.fits.gz"
                else:
                    extracted_fits_path = event_skymap_dir / "skymap.fits"
                
                # Extract member
                if not extract_member(tar, skymap_member, extracted_fits_path, force=args.force):
                    print(f"  Skipping event {event_name} due to extraction failure")
                    continue
        
        except Exception as e:
            print(f"  ERROR: Failed to process tarball: {e}")
            continue
        
        # Plot skymap (optional - add to manifest even if plot fails)
        plot_success = False
        if extracted_fits_path is not None and extracted_fits_path.exists():
            print(f"Step 3: Plot skymap")
            title = f"{event_name} sky probability (2D)"
            plot_success = plot_skymap(extracted_fits_path, figure_path, title)
            if not plot_success:
                print(f"  Plot failed, but FITS file is available")
        
        # Add to manifest (even if plot failed, as long as FITS was extracted)
        if extracted_fits_path is not None and extracted_fits_path.exists():
            manifest.append({
                'event': event_name,
                'sky_area_90': float(sky_area_90) if pd.notna(sky_area_90) else None,
                'skymap_url': skymap_url,
                'tarball_path': str(tarball_path.relative_to(project_root)),
                'extracted_fits_path': str(extracted_fits_path.relative_to(project_root)),
                'figure_path': str(figure_path.relative_to(project_root)) if plot_success else None
            })
    
    # Save manifest
    print(f"\n{'='*60}")
    print(f"Saving manifest to: {manifest_path}")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nDone! Processed {len(manifest)} events successfully.")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()

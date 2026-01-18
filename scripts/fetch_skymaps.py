"""Download and extract skymaps for top-N localized GW events."""

import argparse
import json
import tarfile
from pathlib import Path
from typing import Optional
import pandas as pd
import requests


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
    import numpy as np
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


def load_manifest(manifest_path: Path) -> list:
    """Load existing manifest or return empty list."""
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            return json.load(f)
    return []


def save_manifest(manifest: list, manifest_path: Path) -> None:
    """Save manifest to JSON file."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def update_manifest_entry(manifest: list, event_name: str, 
                          sky_area_90: float, skymap_url: str,
                          tarball_path: str, fits_path: str) -> None:
    """Update or add manifest entry for an event."""
    # Find existing entry
    for entry in manifest:
        if entry.get('event') == event_name:
            entry['sky_area_90'] = sky_area_90
            entry['skymap_url'] = skymap_url
            entry['tarball_path'] = tarball_path
            entry['fits_path'] = fits_path
            return
    
    # Add new entry
    manifest.append({
        'event': event_name,
        'sky_area_90': sky_area_90,
        'skymap_url': skymap_url,
        'tarball_path': tarball_path,
        'fits_path': fits_path
    })


def main():
    """Main function to download and extract skymaps."""
    parser = argparse.ArgumentParser(
        description="Download and extract skymaps for top-N localized GW events"
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
    manifest_path = skymap_dir / "manifest.json"
    
    # Create directories
    tarball_dir.mkdir(parents=True, exist_ok=True)
    skymap_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    # Load existing manifest
    manifest = load_manifest(manifest_path)
    
    # Process each event
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
        fits_path = None
        
        # Download tarball
        print(f"Step 1: Download tarball")
        if not download_tarball(skymap_url, tarball_path, force=args.force):
            print(f"  Skipping event {event_name} due to download failure")
            continue
        
        # Extract skymap FITS
        print(f"Step 2: Extract skymap FITS")
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
                    fits_path = event_skymap_dir / "skymap.fits.gz"
                else:
                    fits_path = event_skymap_dir / "skymap.fits"
                
                # Extract member
                if not extract_member(tar, skymap_member, fits_path, force=args.force):
                    print(f"  Skipping event {event_name} due to extraction failure")
                    continue
        
        except Exception as e:
            print(f"  ERROR: Failed to process tarball: {e}")
            continue
        
        # Update manifest
        if fits_path is not None and fits_path.exists():
            update_manifest_entry(
                manifest, event_name,
                float(sky_area_90) if pd.notna(sky_area_90) else None,
                skymap_url,
                str(tarball_path.relative_to(project_root)),
                str(fits_path.relative_to(project_root))
            )
            print(f"  Updated manifest entry for {event_name}")
    
    # Save manifest
    print(f"\n{'='*60}")
    print(f"Saving manifest to: {manifest_path}")
    save_manifest(manifest, manifest_path)
    
    print(f"\nDone! Processed {len(df_selected)} events.")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()

"""Orchestrate the full pipeline to produce the first H0 posterior plot."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict
import numpy as np
import pandas as pd


def ensure_directories(project_root: Path) -> None:
    """Ensure required directories exist."""
    dirs = [
        project_root / "data",
        project_root / "data" / "skymaps",
        project_root / "data" / "zenodo_tarballs",
        project_root / "figures",
        project_root / "figures" / "skymaps"
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories ensured")


def ensure_events_csv(project_root: Path, force: bool = False) -> bool:
    """Ensure events CSV exists, create if missing."""
    csv_path = project_root / "data" / "events_gwtc3_confident.csv"
    
    if csv_path.exists() and not force:
        print(f"✓ Events CSV exists: {csv_path}")
        return True
    
    print(f"Creating events CSV...")
    try:
        result = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "list_events.py"), 
             "--max-events", "50"],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        if result.returncode == 0 and csv_path.exists():
            print(f"✓ Events CSV created: {csv_path}")
            return True
        else:
            print(f"✗ Failed to create events CSV")
            if result.stderr:
                print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error creating events CSV: {e}")
        return False


def ensure_skymaps_fetched(project_root: Path, top_n: int, force: bool = False) -> bool:
    """Ensure skymaps are fetched for top-N events (incremental: only new events)."""
    manifest_path = project_root / "data" / "skymaps" / "manifest.json"
    
    # Check if we have enough events in manifest
    existing_count = 0
    if manifest_path.exists():
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        existing_count = len(manifest)
        if existing_count >= top_n and not force:
            print(f"✓ Manifest has {existing_count} events (need {top_n})")
            return True
    
    if existing_count > 0:
        print(f"Fetching additional skymaps: {existing_count} -> {top_n} events...")
    else:
        print(f"Fetching skymaps for top-{top_n} events...")
    
    try:
        result = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "fetch_skymaps.py"),
             "--top-n", str(top_n)] + (["--force"] if force else []),
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            # Check final count
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    manifest = json.load(f)
                print(f"✓ Skymaps fetched: {len(manifest)} events in manifest")
            else:
                print(f"✓ Skymaps fetched")
            return True
        else:
            print(f"✗ Failed to fetch skymaps: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error fetching skymaps: {e}")
        return False


def load_manifest(project_root: Path) -> list:
    """Load manifest from JSON file."""
    manifest_path = project_root / "data" / "skymaps" / "manifest.json"
    if not manifest_path.exists():
        return []
    with open(manifest_path, 'r') as f:
        return json.load(f)


def ensure_distance_pdfs(project_root: Path, force: bool = False) -> bool:
    """Ensure distance PDFs are created for events in manifest."""
    manifest = load_manifest(project_root)
    
    if len(manifest) == 0:
        print(f"✗ Manifest is empty")
        return False
    
    # Check which events have distance PDFs
    events_with_pdf = []
    events_without_pdf = []
    
    for entry in manifest:
        event_name = entry.get('event')
        pdf_path_rel = entry.get('distance_pdf_path')
        if pdf_path_rel:
            pdf_path = project_root / pdf_path_rel
            if pdf_path.exists():
                events_with_pdf.append(event_name)
            else:
                events_without_pdf.append(event_name)
        else:
            events_without_pdf.append(event_name)
    
    if len(events_with_pdf) > 0 and not force:
        print(f"✓ Found {len(events_with_pdf)} events with distance PDFs")
        if len(events_without_pdf) > 0:
            print(f"  ({len(events_without_pdf)} events still need distance PDFs)")
        return True
    
    if len(events_without_pdf) == 0:
        print(f"✓ All events have distance PDFs")
        return True
    
    print(f"Creating distance PDFs for {len(events_without_pdf)} events...")
    try:
        result = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "plot_skymaps.py"),
             "--distance", "--area"] + (["--force"] if force else []),
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✓ Distance PDFs created")
            return True
        else:
            print(f"✗ Failed to create distance PDFs: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ Error creating distance PDFs: {e}")
        return False


def find_event_with_3d_distance(
    project_root: Path,
    top_n_start: int,
    top_n_max: int,
    top_n_step: int,
    max_attempts: int,
    prefer_area_from_map: bool,
    force: bool = False
) -> Optional[Dict]:
    """
    Escalate top-n until an event with 3D distance is found.
    
    Returns the best candidate event, or None if none found after max_attempts.
    """
    print(f"\n{'='*60}")
    print(f"Escalation Loop: Finding Event with 3D Distance")
    print(f"{'='*60}")
    print(f"  Start: top_n={top_n_start}, Max: {top_n_max}, Step: {top_n_step}")
    print(f"  Max attempts: {max_attempts}")
    print(f"  Prefer area from map: {prefer_area_from_map}")
    print()
    
    for attempt in range(1, max_attempts + 1):
        # Calculate current top_n
        top_n = min(top_n_start + (attempt - 1) * top_n_step, top_n_max)
        
        print(f"Attempt {attempt}/{max_attempts}: top_n={top_n}")
        print(f"{'-'*60}")
        
        # Step 1: Fetch skymaps (incremental)
        if not ensure_skymaps_fetched(project_root, top_n, force=force):
            print(f"  ✗ Failed to fetch skymaps, continuing...")
            continue
        
        # Step 2: Extract distance PDFs
        if not ensure_distance_pdfs(project_root, force=force):
            print(f"  ⚠ Distance PDF extraction had issues, continuing...")
        
        # Step 3: Reload manifest and check for candidates
        manifest = load_manifest(project_root)
        total, with_fits, with_distance_pdf = count_events_with_3d_distance(project_root, manifest)
        
        print(f"  Manifest status:")
        print(f"    Total events: {total}")
        print(f"    With FITS: {with_fits}")
        print(f"    With distance PDF: {with_distance_pdf}")
        
        # Step 4: Try to find a candidate
        best_event = choose_best_event(project_root, prefer_area_from_map=prefer_area_from_map)
        
        if best_event is not None:
            print(f"  ✓ Found candidate: {best_event['event']} (sky_area={best_event['sky_area']:.1f} deg²)")
            return best_event
        else:
            print(f"  ✗ No 3D distance events found yet at top_n={top_n}")
            if attempt < max_attempts:
                print(f"  → Escalating to next attempt...")
            print()
    
    # No event found after all attempts
    return None


def count_events_with_3d_distance(project_root: Path, manifest: list) -> tuple[int, int, int]:
    """
    Count events in manifest with different statuses.
    
    Returns:
        (total, with_fits, with_distance_pdf)
    """
    total = len(manifest)
    with_fits = 0
    with_distance_pdf = 0
    
    for entry in manifest:
        fits_path_rel = entry.get('fits_path') or entry.get('extracted_fits_path')
        pdf_path_rel = entry.get('distance_pdf_path')
        
        if fits_path_rel:
            fits_path = project_root / fits_path_rel
            if fits_path.exists():
                with_fits += 1
        
        if pdf_path_rel:
            pdf_path = project_root / pdf_path_rel
            if pdf_path.exists():
                with_distance_pdf += 1
    
    return total, with_fits, with_distance_pdf


def choose_best_event(project_root: Path, prefer_area_from_map: bool = True) -> Optional[Dict]:
    """
    Choose the best event for inference (smallest sky_area_90).
    
    Parameters
    ----------
    prefer_area_from_map : bool
        If True, prefer sky_area_90_from_map over sky_area_90 when both exist.
    """
    manifest = load_manifest(project_root)
    
    if len(manifest) == 0:
        return None
    
    candidates = []
    
    for entry in manifest:
        event_name = entry.get('event')
        fits_path_rel = entry.get('fits_path') or entry.get('extracted_fits_path')
        pdf_path_rel = entry.get('distance_pdf_path')
        
        # Must have both FITS and distance PDF
        if not fits_path_rel or not pdf_path_rel:
            continue
        
        fits_path = project_root / fits_path_rel
        pdf_path = project_root / pdf_path_rel
        
        if not (fits_path.exists() and pdf_path.exists()):
            continue
        
        # Get sky area (prefer from_map if requested, fallback to catalog value)
        if prefer_area_from_map:
            sky_area = entry.get('sky_area_90_from_map') or entry.get('sky_area_90')
        else:
            sky_area = entry.get('sky_area_90') or entry.get('sky_area_90_from_map')
        
        if sky_area is None or not np.isfinite(sky_area):
            continue
        
        candidates.append({
            'event': event_name,
            'sky_area': float(sky_area),
            'fits_path': fits_path_rel,
            'pdf_path': pdf_path_rel,
            'entry': entry
        })
    
    if len(candidates) == 0:
        return None
    
    # Sort by sky_area (smallest first) - deterministic
    candidates.sort(key=lambda x: (x['sky_area'], x['event']))
    
    best = candidates[0]
    return best


def ensure_galaxy_catalog(project_root: Path, glade_raw: Optional[str] = None) -> bool:
    """Ensure galaxy catalog exists, create if needed."""
    parquet_path = project_root / "data" / "glade_plus_min.parquet"
    
    if parquet_path.exists():
        print(f"✓ Galaxy catalog exists: {parquet_path}")
        return True
    
    if glade_raw is None:
        print(f"\n{'='*60}")
        print(f"ERROR: Galaxy catalog not found: {parquet_path}")
        print(f"{'='*60}")
        print(f"\nPlease download GLADE+ catalog and provide the path:")
        print(f"  python scripts/run_first_result.py --glade-raw <path_to_glade_file>")
        print(f"\nOr create the minimal catalog manually:")
        print(f"  python scripts/ingest_glade_plus_min.py <raw_glade_file>")
        return False
    
    # Ingest GLADE+ file
    print(f"Ingesting GLADE+ from: {glade_raw}")
    return ingest_glade_plus(Path(glade_raw), parquet_path)


def ingest_glade_plus(raw_path: Path, out_path: Path) -> bool:
    """
    Create minimal GLADE+ parquet from raw file.
    
    Supports common GLADE+ formats (CSV, FITS, Parquet).
    Maps common column names (RA, DEC, z) to required names (ra_deg, dec_deg, z).
    """
    if not raw_path.exists():
        print(f"✗ GLADE+ file not found: {raw_path}")
        return False
    
    print(f"  Reading GLADE+ file...")
    
    try:
        # Try different formats
        if raw_path.suffix == '.fits' or raw_path.suffix == '.fit':
            from astropy.io import fits
            with fits.open(str(raw_path)) as hdul:
                hdu = hdul[1]  # Usually data is in HDU 1
                df = pd.DataFrame(hdu.data)
        elif raw_path.suffix == '.parquet':
            df = pd.read_parquet(raw_path)
        else:
            # Assume CSV or similar
            df = pd.read_csv(raw_path, low_memory=False)
        
        print(f"  Loaded {len(df)} rows")
        
        # Map common column names to our required names
        column_mapping = {
            # Common GLADE+ column names
            'RA': 'ra_deg',
            'ra': 'ra_deg',
            'RAJ2000': 'ra_deg',
            'DEC': 'dec_deg',
            'dec': 'dec_deg',
            'DEJ2000': 'dec_deg',
            'z': 'z',
            'zhelio': 'z',
            'redshift': 'z',
            'z_helio': 'z',
            'weight': 'weight',
            'w': 'weight'
        }
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Check required columns
        required = ['ra_deg', 'dec_deg', 'z']
        missing = [col for col in required if col not in df.columns]
        if missing:
            print(f"✗ Missing required columns after mapping: {missing}")
            print(f"  Available columns: {list(df.columns)}")
            return False
        
        # Add weight if missing
        if 'weight' not in df.columns:
            df['weight'] = 1.0
        
        # Filter to valid data
        df = df[
            df['ra_deg'].notna() & df['dec_deg'].notna() & df['z'].notna() &
            (df['z'] > 0) & (df['z'] < 10) &
            np.isfinite(df['ra_deg']) & np.isfinite(df['dec_deg']) & np.isfinite(df['z'])
        ].copy()
        
        if len(df) == 0:
            print(f"✗ No valid galaxies after filtering")
            return False
        
        # Select only needed columns
        df_min = df[['ra_deg', 'dec_deg', 'z', 'weight']].copy()
        
        # Save to parquet
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_min.to_parquet(out_path, index=False)
        
        print(f"✓ Created minimal catalog: {out_path} ({len(df_min)} galaxies)")
        return True
        
    except Exception as e:
        print(f"✗ Failed to ingest GLADE+: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_h0_inference(project_root: Path, event_name: str, 
                    h0_min: float, h0_max: float, h0_n: int,
                    kappa: float, galaxy_path: str) -> bool:
    """Run H0 inference for the selected event."""
    print(f"\n{'='*60}")
    print(f"Running H0 inference for: {event_name}")
    print(f"{'='*60}")
    
    out_path = project_root / "figures" / f"first_h0_post_{event_name}.png"
    
    try:
        result = subprocess.run(
            [sys.executable, str(project_root / "scripts" / "one_event_h0.py"),
             "--event", event_name,
             "--model", "both",
             "--kappa", str(kappa),
             "--h0-min", str(h0_min),
             "--h0-max", str(h0_max),
             "--h0-n", str(h0_n),
             "--galaxies", galaxy_path,
             "--out", str(out_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ H0 inference completed")
            if out_path.exists():
                return True
            else:
                print(f"✗ Output file not created: {out_path}")
                return False
        else:
            print(f"✗ H0 inference failed:")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"✗ Error running H0 inference: {e}")
        return False


def main():
    """Main orchestration function."""
    parser = argparse.ArgumentParser(
        description="Orchestrate full pipeline to produce first H0 posterior plot"
    )
    parser.add_argument(
        '--top-n-start',
        type=int,
        default=5,
        help='Initial number of top events to fetch (default: 5)'
    )
    parser.add_argument(
        '--top-n-max',
        type=int,
        default=60,
        help='Maximum number of events to try (default: 60)'
    )
    parser.add_argument(
        '--top-n-step',
        type=int,
        default=5,
        help='Increment step for top-n escalation (default: 5)'
    )
    parser.add_argument(
        '--max-attempts',
        type=int,
        default=8,
        help='Maximum escalation attempts (default: 8)'
    )
    parser.add_argument(
        '--prefer-area-from-map',
        action='store_true',
        default=True,
        help='Prefer sky_area_90_from_map over sky_area_90 (default: True)'
    )
    parser.add_argument(
        '--no-prefer-area-from-map',
        dest='prefer_area_from_map',
        action='store_false',
        help='Do not prefer sky_area_90_from_map'
    )
    parser.add_argument(
        '--kappa',
        type=float,
        default=1.0,
        help='kappa parameter for ell model (default: 1.0)'
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
        '--glade-raw',
        type=str,
        default=None,
        help='Path to raw GLADE+ file (will be ingested if provided)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-fetch and re-plot even if cached'
    )
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    print(f"{'='*60}")
    print(f"Pipeline Orchestration: First H0 Result")
    print(f"{'='*60}\n")
    
    # Step A: Ensure directories
    print("Step A: Ensure directories")
    ensure_directories(project_root)
    
    # Step B: Ensure events CSV
    print(f"\nStep B: Ensure events CSV")
    if not ensure_events_csv(project_root, force=args.force):
        print(f"ERROR: Failed to create events CSV")
        return 1
    
    # Step C-D-E: Escalation loop to find event with 3D distance
    best_event = find_event_with_3d_distance(
        project_root,
        top_n_start=args.top_n_start,
        top_n_max=args.top_n_max,
        top_n_step=args.top_n_step,
        max_attempts=args.max_attempts,
        prefer_area_from_map=args.prefer_area_from_map,
        force=args.force
    )
    
    if best_event is None:
        print(f"\n{'='*60}")
        print(f"ERROR: No event with 3D distance found after {args.max_attempts} attempts")
        print(f"{'='*60}")
        print(f"\nPossible solutions:")
        print(f"  1. Increase --top-n-max (current: {args.top_n_max})")
        print(f"  2. Some GWTC-3 releases provide only 2D skymaps (no distance info)")
        print(f"  3. Try installing ligo-skymap in WSL/Linux for better distance handling:")
        print(f"     pip install ligo-skymap")
        print(f"  4. Check if distance PDFs were created:")
        print(f"     ls data/skymaps/*/distance_pdf.npz")
        return 1
    
    event_name = best_event['event']
    print(f"\n{'='*60}")
    print(f"Selected Event: {event_name}")
    print(f"{'='*60}")
    print(f"  Sky area: {best_event['sky_area']:.1f} deg²")
    print(f"  FITS: {best_event['fits_path']}")
    print(f"  Distance PDF: {best_event['pdf_path']}")
    
    # Step F: Ensure galaxy catalog
    print(f"\nStep F: Ensure galaxy catalog")
    galaxy_path = "data/glade_plus_min.parquet"
    if not ensure_galaxy_catalog(project_root, args.glade_raw):
        return 1
    
    # Step G: Run inference
    print(f"\nStep G: Run H0 inference")
    if not run_h0_inference(
        project_root, event_name,
        args.h0_min, args.h0_max, args.h0_n,
        args.kappa, galaxy_path
    ):
        print(f"ERROR: H0 inference failed")
        return 1
    
    # Step H: Print success message
    print(f"\n{'='*60}")
    print(f"SUCCESS: First H0 Result Generated")
    print(f"{'='*60}\n")
    
    print(f"Event: {event_name}")
    print(f"  FITS: {best_event['fits_path']}")
    print(f"  Distance PDF: {best_event['pdf_path']}")
    print(f"  Galaxy catalog: {galaxy_path}")
    print(f"  Output figure: figures/first_h0_post_{event_name}.png")
    print(f"\nSky area: {best_event['sky_area']:.1f} deg²")
    print(f"\nDone!")


if __name__ == "__main__":
    sys.exit(main())

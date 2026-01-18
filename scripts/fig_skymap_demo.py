"""Generate skymap demonstration figure for a GW event."""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from epjc_h0.gwosc_io import (
    get_event_json,
    extract_links,
    choose_best_skymap_url,
    download
)
from epjc_h0.plots import plot_skymap_healpix


def main():
    """Generate skymap demonstration figure."""
    parser = argparse.ArgumentParser(
        description="Download and plot GW event skymap"
    )
    parser.add_argument(
        "event",
        nargs="?",
        default="GW200311_115853",
        help="GW event name (default: GW200311_115853)"
    )
    args = parser.parse_args()
    
    event_name = args.event
    
    # Paths
    project_root = Path(__file__).parent.parent
    skymap_dir = project_root / "data" / "skymaps"
    skymap_path = skymap_dir / f"{event_name}.fits.gz"
    figure_path = project_root / "figures" / f"skymap_{event_name}.png"
    
    # Check if skymap already exists
    if skymap_path.exists():
        print(f"Using existing skymap: {skymap_path}")
    else:
        print(f"Fetching event data for {event_name}...")
        try:
            event_json = get_event_json(event_name)
            print("Event JSON fetched successfully")
            
            # Extract skymap URLs
            skymap_urls, posterior_urls = extract_links(event_json, event_name)
            print(f"Found {len(skymap_urls)} skymap URL(s)")
            
            if not skymap_urls:
                print("ERROR: No skymap URLs found in event JSON")
                print("Available keys in event JSON:", list(event_json.keys()) if isinstance(event_json, dict) else "N/A")
                return
            
            # Choose best skymap URL
            best_url = choose_best_skymap_url(skymap_urls)
            if best_url is None:
                print("ERROR: No suitable skymap URL found")
                return
            
            print(f"Downloading skymap from: {best_url}")
            download(best_url, skymap_path)
            print(f"Skymap saved to: {skymap_path}")
        
        except Exception as e:
            print(f"ERROR: Failed to fetch/download skymap: {e}")
            return
    
    # Generate plot
    print(f"Generating skymap plot...")
    try:
        plot_skymap_healpix(skymap_path, figure_path, 
                           title=f"Skymap: {event_name}")
        print(f"Figure saved to: {figure_path}")
    except Exception as e:
        print(f"ERROR: Failed to generate plot: {e}")
        return


if __name__ == "__main__":
    main()

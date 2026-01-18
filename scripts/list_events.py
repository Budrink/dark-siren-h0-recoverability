"""List GWTC-3-confident events and generate CSV table with metadata."""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from gwosc import api


def safe_get_nested(data: Any, *keys: str, default: Any = None) -> Any:
    """
    Safely extract nested dictionary values.
    
    Parameters
    ----------
    data : Any
        Dictionary or nested structure
    *keys : str
        Sequence of keys to traverse
    default : Any, optional
        Default value if key path not found, by default None
    
    Returns
    -------
    Any
        Value at nested path or default
    """
    current = data
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


def list_events_from_catalog(catalog_json: Union[Dict, List]) -> tuple:
    """
    Extract event names and event data from catalog JSON.
    
    Handles common schemas:
    - Top-level dict with key "events" (dict) -> use its keys and values
    - List of events -> extract event names/identifiers
    
    Parameters
    ----------
    catalog_json : Union[Dict, List]
        Catalog JSON from GWOSC API
    
    Returns
    -------
    tuple[List[str], Dict[str, Dict]]
        (List of event names, Dict mapping event names to their data)
    """
    event_names = []
    event_data = {}
    
    if isinstance(catalog_json, dict):
        # Check for "events" key
        if "events" in catalog_json:
            events = catalog_json["events"]
            if isinstance(events, dict):
                # Events is a dict, use its keys and store data
                for event_key, event_info in events.items():
                    event_names.append(event_key)
                    event_data[event_key] = event_info
            elif isinstance(events, list):
                # Events is a list, extract names/identifiers
                for event in events:
                    if isinstance(event, str):
                        event_names.append(event)
                    elif isinstance(event, dict):
                        # Try common keys for event name
                        name = (event.get("name") or 
                               event.get("event") or 
                               event.get("shortName") or
                               event.get("id"))
                        if name:
                            event_names.append(name)
                            event_data[name] = event
        else:
            # Try to use dict keys directly if it looks like event names
            for key, value in catalog_json.items():
                if isinstance(key, str) and key.startswith("GW"):
                    event_names.append(key)
                    if isinstance(value, dict):
                        event_data[key] = value
    
    elif isinstance(catalog_json, list):
        # Direct list of events
        for event in catalog_json:
            if isinstance(event, str):
                event_names.append(event)
            elif isinstance(event, dict):
                name = (event.get("name") or 
                       event.get("event") or 
                       event.get("shortName") or
                       event.get("id"))
                if name:
                    event_names.append(name)
                    event_data[name] = event
    
    # Remove duplicates and sort
    event_names = sorted(list(set(event_names)))
    return event_names, event_data


def has_skymap(links: Union[Dict, List, None]) -> bool:
    """
    Check if skymap is available in links/files structure.
    
    Detects skymaps by:
    - description contains "skymap" or "bayestar"
    - url ends with .fits/.fits.gz/.multiorder.fits(.gz)
    - key name contains "skymap" (e.g., "files.skymap")
    
    Parameters
    ----------
    links : Union[Dict, List, None]
        Links/files structure from event JSON (can be dict, list, or None)
    
    Returns
    -------
    bool
        True if skymap detected, False otherwise
    """
    if links is None:
        return False
    
    def check_item(item: Any) -> bool:
        """Check a single link item."""
        if isinstance(item, str):
            # Direct URL string
            url_lower = item.lower()
            return (url_lower.endswith('.fits') or 
                    url_lower.endswith('.fits.gz') or
                    '.multiorder.fits' in url_lower or
                    '.multiorder.fits.gz' in url_lower)
        elif isinstance(item, dict):
            # Dictionary with url/description
            url = item.get('url', '')
            desc = item.get('description', '').lower()
            
            url_lower = url.lower()
            has_fits = (url_lower.endswith('.fits') or 
                       url_lower.endswith('.fits.gz') or
                       '.multiorder.fits' in url_lower or
                       '.multiorder.fits.gz' in url_lower)
            
            has_skymap_desc = ('skymap' in desc or 'bayestar' in desc)
            
            return has_fits or has_skymap_desc
        
        return False
    
    # Handle list of items
    if isinstance(links, list):
        return any(check_item(item) for item in links)
    
    # Handle dictionary
    if isinstance(links, dict):
        # Check if any key contains "skymap"
        for key in links.keys():
            if isinstance(key, str) and 'skymap' in key.lower():
                return True
        
        # Check all values recursively
        for value in links.values():
            if isinstance(value, (list, dict)):
                if has_skymap(value):
                    return True
            elif check_item(value):
                return True
    
    return False


def fetch_event_json_robust(event_name: str, 
                            catalog: str = "GWTC-3-confident",
                            version: str = "v1") -> Optional[Dict]:
    """
    Fetch event JSON with fallback to direct REST API.
    
    Parameters
    ----------
    event_name : str
        Event name (may include version suffix like "-v1")
    catalog : str, optional
        Catalog name, by default "GWTC-3-confident"
    version : str, optional
        API version, by default "v1"
    
    Returns
    -------
    Optional[Dict]
        Event JSON data or None if both methods fail
    """
    # Clean event name (remove version suffix if present)
    event_name_clean = event_name.split('-v')[0] if '-v' in event_name else event_name
    
    # Step 1: Try gwosc API
    try:
        event_json = api.fetch_event_json(
            event_name_clean,
            catalog=catalog,
            version=version,
            host="https://gwosc.org"
        )
        # Check if we got a valid structure
        if isinstance(event_json, dict):
            return event_json
    except Exception:
        pass
    
    # Step 2: Fallback to direct REST API
    try:
        url = f"https://gwosc.org/eventapi/json/{catalog}/{event_name_clean}/{version}/"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        event_json = response.json()
        if isinstance(event_json, dict):
            return event_json
    except Exception:
        pass
    
    return None


def extract_preferred_pe_parameters(event_json: Dict, event_name: str) -> Optional[Dict]:
    """
    Extract preferred PE (parameter estimation) parameters from event JSON.
    
    Looks for parameters inside event_json["events"][<event_uid>]["parameters"].
    Prefers entries with pipeline_type="pe" and is_preferred=True.
    
    Parameters
    ----------
    event_json : Dict
        Event JSON data from GWOSC API
    event_name : str
        Event name (may include version suffix like "-v1")
    
    Returns
    -------
    Optional[Dict]
        Preferred PE parameters dict or None if not found
    """
    if not isinstance(event_json, dict) or "events" not in event_json:
        return None
    
    events = event_json["events"]
    if not isinstance(events, dict):
        return None
    
    # Try to find event data by event_name or cleaned name
    event_data = None
    event_name_clean = event_name.split('-v')[0] if '-v' in event_name else event_name
    
    # Try full event name first
    if event_name in events:
        event_data = events[event_name]
    # Try cleaned name
    elif event_name_clean in events:
        event_data = events[event_name_clean]
    # If still not found, take first event
    elif len(events) > 0:
        event_data = list(events.values())[0]
    
    if not isinstance(event_data, dict) or "parameters" not in event_data:
        return None
    
    parameters = event_data["parameters"]
    if not isinstance(parameters, dict):
        return None
    
    # Filter for PE pipeline type
    pe_params = []
    for param_name, param_data in parameters.items():
        if isinstance(param_data, dict):
            pipeline_type = param_data.get("pipeline_type", "").lower()
            if pipeline_type == "pe":
                pe_params.append((param_name, param_data))
    
    if not pe_params:
        return None
    
    # Prefer is_preferred=True, else take first
    preferred_param = None
    for param_name, param_data in pe_params:
        if param_data.get("is_preferred") is True:
            preferred_param = param_data
            break
    
    # Fallback to first PE entry
    if preferred_param is None:
        preferred_param = pe_params[0][1]
    
    return preferred_param


def extract_event_rows(event_names: List[str],
                       catalog_data: Dict[str, Dict],
                       catalog: str = "GWTC-3-confident",
                       version: str = "v1",
                       max_events: Optional[int] = None) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Extract metadata rows for each event.
    
    Parameters
    ----------
    event_names : List[str]
        List of event names to process
    catalog_data : Dict[str, Dict]
        Dictionary mapping event names to their catalog data
    catalog : str, optional
        Catalog name, by default "GWTC-3-confident"
    version : str, optional
        API version for individual events, by default "v1"
    max_events : Optional[int], optional
        Maximum number of events to process, by default None (all)
    
    Returns
    -------
    tuple[List[Dict[str, Any]], Dict[str, int]]
        (List of event metadata dictionaries, statistics dict)
    """
    rows = []
    stats = {
        'successful_json': 0,
        'skymaps_detected': 0
    }
    
    # Limit events if specified
    events_to_process = event_names[:max_events] if max_events else event_names
    
    for event_name in events_to_process:
        # Get data from catalog (already available)
        catalog_event = catalog_data.get(event_name, {})
        
        # Extract basic info from catalog data
        gps_time = catalog_event.get('GPS') or catalog_event.get('gps_time') or catalog_event.get('gps')
        
        # Network SNR from catalog
        network_snr = (catalog_event.get('network_matched_filter_snr') or 
                      catalog_event.get('network_snr') or
                      catalog_event.get('snr'))
        
        # FAR from catalog
        far = catalog_event.get('far') or catalog_event.get('FAR')
        
        # Luminosity distance from catalog
        dL_median = catalog_event.get('luminosity_distance')
        dL_lower = catalog_event.get('luminosity_distance_lower')
        dL_upper = catalog_event.get('luminosity_distance_upper')
        
        # Try to fetch full event JSON for links/skymap info
        skymap_available = False
        skymap_url = ""
        posterior_samples_url = ""
        sky_area_90 = None
        
        # Fetch event JSON with fallback
        event_json = fetch_event_json_robust(event_name, catalog=catalog, version=version)
        
        if event_json is not None:
            stats['successful_json'] += 1
            
            # Extract preferred PE parameters
            pe_params = extract_preferred_pe_parameters(event_json, event_name)
            
            if pe_params is not None:
                # Extract links from PE parameters
                links = pe_params.get("links")
                if isinstance(links, dict):
                    # Check for skymap URL
                    skymap_url_val = links.get("skymap")
                    if skymap_url_val:
                        if isinstance(skymap_url_val, str):
                            skymap_url = skymap_url_val
                        elif isinstance(skymap_url_val, list) and len(skymap_url_val) > 0:
                            skymap_url = skymap_url_val[0] if isinstance(skymap_url_val[0], str) else ""
                        elif isinstance(skymap_url_val, dict) and "url" in skymap_url_val:
                            skymap_url = skymap_url_val["url"]
                        
                        if skymap_url:
                            skymap_available = True
                            stats['skymaps_detected'] += 1
                    
                    # Check for posterior-samples URL
                    posterior_samples_val = links.get("posterior-samples")
                    if posterior_samples_val:
                        if isinstance(posterior_samples_val, str):
                            posterior_samples_url = posterior_samples_val
                        elif isinstance(posterior_samples_val, list) and len(posterior_samples_val) > 0:
                            posterior_samples_url = posterior_samples_val[0] if isinstance(posterior_samples_val[0], str) else ""
                        elif isinstance(posterior_samples_val, dict) and "url" in posterior_samples_val:
                            posterior_samples_url = posterior_samples_val["url"]
                
                # Extract sky_area from PE parameters
                sky_area_val = pe_params.get("sky_area")
                if sky_area_val is not None:
                    try:
                        sky_area_90 = float(sky_area_val)
                    except (ValueError, TypeError):
                        pass
        
        # Create row with all collected data
        row = {
            'event': event_name,
            'gps_time': gps_time if gps_time is not None else np.nan,
            'network_snr': network_snr if network_snr is not None else np.nan,
            'far': far if far is not None else np.nan,
            'luminosity_distance_median': dL_median if dL_median is not None else np.nan,
            'luminosity_distance_lower': dL_lower if dL_lower is not None else np.nan,
            'luminosity_distance_upper': dL_upper if dL_upper is not None else np.nan,
            'sky_area_90': sky_area_90 if sky_area_90 is not None else np.nan,
            'skymap_available': skymap_available,
            'skymap_url': skymap_url,
            'posterior_samples_url': posterior_samples_url
        }
        
        rows.append(row)
    
    return rows, stats


def plot_top_localized(df: pd.DataFrame, out_path: Path, top_n: int = 20) -> None:
    """
    Create bar chart of top N most localized events.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with event data
    out_path : Path
        Output file path for figure
    top_n : int, optional
        Number of top events to show, by default 20
    """
    # Filter events with finite sky_area_90
    df_valid = df[df['sky_area_90'].notna() & np.isfinite(df['sky_area_90'])].copy()
    
    if len(df_valid) < 5:
        print(f"Skipping plot: only {len(df_valid)} events with sky_area_90 data (need at least 5)")
        return
    
    # Sort by sky_area_90 ascending and take top N
    df_top = df_valid.nsmallest(top_n, 'sky_area_90')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create bar chart
    y_pos = np.arange(len(df_top))
    bars = ax.barh(y_pos, df_top['sky_area_90'], align='center')
    
    # Set labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_top['event'], fontsize=9)
    ax.set_xlabel('Sky Area (90% credible region, deg²)', fontsize=12)
    ax.set_title(f'Top {len(df_top)} Most Localized GW Events (GWTC-3-confident)', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Invert y-axis to show best (smallest) at top
    ax.invert_yaxis()
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(df_top.iterrows()):
        ax.text(row['sky_area_90'] + max(df_top['sky_area_90']) * 0.01, 
                i, f"{row['sky_area_90']:.1f}", 
                va='center', fontsize=8)
    
    plt.tight_layout()
    
    # Create output directory if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to: {out_path}")


def main():
    """Main function to generate event list CSV and figure."""
    parser = argparse.ArgumentParser(
        description="List GWTC-3-confident events and generate CSV table with metadata"
    )
    parser.add_argument(
        '--max-events',
        type=int,
        default=35,
        help='Maximum number of events to process (default: 35)'
    )
    parser.add_argument(
        '--no-per-event',
        action='store_true',
        help='Only dump the catalog list quickly without per-event metadata'
    )
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "events_gwtc3_confident.csv"
    figure_path = project_root / "figures" / "top_localized_events.png"
    
    # Create output directories
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("Fetching GWTC-3-confident catalog...")
    try:
        # Use correct API: fetch_catalog_json WITHOUT version
        catalog_json = api.fetch_catalog_json(
            "GWTC-3-confident",
            host="https://gwosc.org"
        )
        
        # Extract event names and catalog data
        event_names, catalog_data = list_events_from_catalog(catalog_json)
        
        if not event_names:
            raise ValueError(
                "No events found in catalog. "
                "Check that the catalog name is correct and the API is accessible."
            )
        
        print(f"Found {len(event_names)} events in catalog")
        
        # If --no-per-event, just save the list
        if args.no_per_event:
            df = pd.DataFrame({'event': event_names})
            df.to_csv(csv_path, index=False)
            print(f"Event list saved to: {csv_path}")
            print(f"Total events: {len(event_names)}")
            return
        
    except Exception as e:
        print(f"ERROR: Failed to fetch catalog: {e}")
        raise
    
    # Process events with metadata
    print(f"Processing up to {args.max_events} events for metadata...")
    rows, stats = extract_event_rows(
        event_names,
        catalog_data,
        catalog="GWTC-3-confident", 
        version="v1",
        max_events=args.max_events
    )
    
    if not rows:
        raise ValueError(
            "No event data extracted. "
            "Check API connectivity and event data availability."
        )
    
    # Log statistics
    print(f"\nPer-event JSON retrieval statistics:")
    print(f"  Successful per-event JSON: {stats['successful_json']} / {len(rows)}")
    print(f"  Skymaps detected: {stats['skymaps_detected']} / {len(rows)}")
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Sort: primary by sky_area_90 ascending (NaNs last), secondary by luminosity_distance_median ascending
    df_sorted = df.copy()
    df_sorted['_sort_sky'] = df_sorted['sky_area_90'].fillna(np.inf)
    df_sorted['_sort_dL'] = df_sorted['luminosity_distance_median'].fillna(np.inf)
    df_sorted = df_sorted.sort_values(['_sort_sky', '_sort_dL'], ascending=[True, True])
    df_sorted = df_sorted.drop(columns=['_sort_sky', '_sort_dL'])
    
    # Save CSV
    df_sorted.to_csv(csv_path, index=False)
    print(f"CSV saved to: {csv_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total events processed: {len(df_sorted)}")
    print(f"Events with skymaps: {df_sorted['skymap_available'].sum()}")
    print(f"Events with sky_area_90: {df_sorted['sky_area_90'].notna().sum()}")
    print(f"Events with luminosity_distance: {df_sorted['luminosity_distance_median'].notna().sum()}")
    
    # Show top 20 by sky_area_90
    df_valid_sky = df_sorted[df_sorted['sky_area_90'].notna() & 
                             np.isfinite(df_sorted['sky_area_90'])].head(20)
    if len(df_valid_sky) > 0:
        print(f"\nTop {len(df_valid_sky)} events by sky_area_90 (deg²):")
        print("-" * 60)
        for idx, row in df_valid_sky.iterrows():
            dL_str = f"{row['luminosity_distance_median']:.1f}" if pd.notna(row['luminosity_distance_median']) else "N/A"
            skymap_str = "✓" if row['skymap_available'] else "✗"
            print(f"  {row['event']:20s} | {row['sky_area_90']:8.2f} deg² | dL={dL_str:>8s} Mpc | skymap: {skymap_str}")
    
    # Generate plot
    print("\nGenerating localization plot...")
    plot_top_localized(df_sorted, figure_path, top_n=20)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

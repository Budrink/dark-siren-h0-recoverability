"""GWOSC API I/O utilities for fetching event data and skymaps."""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
from gwosc import api


def get_event_json(event: str, catalog: str = "GWTC-3-confident", 
                   version: str = "v1") -> Dict:
    """
    Fetch event JSON from GWOSC API.
    
    Parameters
    ----------
    event : str
        Event name (e.g., "GW200311_115853")
    catalog : str, optional
        Catalog name, by default "GWTC-3-confident"
    version : str, optional
        API version, by default "v1"
    
    Returns
    -------
    Dict
        Event JSON data
    """
    return api.fetch_event_json(event, catalog=catalog, version=version)


def extract_links(event_json: Dict, event_name: str) -> Tuple[List[str], List[str]]:
    """
    Extract skymap and posterior URLs from event JSON.
    
    Handles common schema variants in GWOSC event data.
    
    Parameters
    ----------
    event_json : Dict
        Event JSON data from GWOSC API
    event_name : str
        Event name for logging/debugging
    
    Returns
    -------
    Tuple[List[str], List[str]]
        (skymap_urls, posterior_urls) lists
    """
    skymap_urls = []
    posterior_urls = []
    
    # Try different possible locations in the JSON structure
    if isinstance(event_json, dict):
        # Check for direct 'files' key
        if 'files' in event_json:
            files = event_json['files']
            if isinstance(files, dict):
                # Common structure: files -> skymap -> ...
                if 'skymap' in files:
                    skymap_data = files['skymap']
                    if isinstance(skymap_data, dict):
                        # Could be a dict with URL or list of URLs
                        if 'url' in skymap_data:
                            skymap_urls.append(skymap_data['url'])
                        elif isinstance(skymap_data, list):
                            for item in skymap_data:
                                if isinstance(item, dict) and 'url' in item:
                                    skymap_urls.append(item['url'])
                    elif isinstance(skymap_data, list):
                        for item in skymap_data:
                            if isinstance(item, str):
                                skymap_urls.append(item)
                            elif isinstance(item, dict) and 'url' in item:
                                skymap_urls.append(item['url'])
                
                # Check for posterior files
                if 'posterior' in files:
                    posterior_data = files['posterior']
                    if isinstance(posterior_data, dict) and 'url' in posterior_data:
                        posterior_urls.append(posterior_data['url'])
                    elif isinstance(posterior_data, list):
                        for item in posterior_data:
                            if isinstance(item, str):
                                posterior_urls.append(item)
                            elif isinstance(item, dict) and 'url' in item:
                                posterior_urls.append(item['url'])
        
        # Alternative: check for 'skymap' at top level
        if 'skymap' in event_json:
            skymap_val = event_json['skymap']
            if isinstance(skymap_val, str):
                skymap_urls.append(skymap_val)
            elif isinstance(skymap_val, list):
                skymap_urls.extend([url for url in skymap_val if isinstance(url, str)])
        
        # Alternative: check for 'urls' key
        if 'urls' in event_json:
            urls = event_json['urls']
            if isinstance(urls, dict):
                if 'skymap' in urls:
                    skymap_val = urls['skymap']
                    if isinstance(skymap_val, str):
                        skymap_urls.append(skymap_val)
                    elif isinstance(skymap_val, list):
                        skymap_urls.extend([url for url in skymap_val if isinstance(url, str)])
    
    # Remove duplicates while preserving order
    seen = set()
    skymap_urls = [url for url in skymap_urls if url not in seen and not seen.add(url)]
    seen = set()
    posterior_urls = [url for url in posterior_urls if url not in seen and not seen.add(url)]
    
    return skymap_urls, posterior_urls


def choose_best_skymap_url(skymap_urls: List[str]) -> Optional[str]:
    """
    Choose the best skymap URL from a list, preferring FITS files.
    
    Parameters
    ----------
    skymap_urls : List[str]
        List of skymap URLs
    
    Returns
    -------
    Optional[str]
        Best skymap URL, or None if no suitable URL found
    """
    # Prefer .fits.gz, then .fits
    for url in skymap_urls:
        if url.endswith('.fits.gz') or url.endswith('.fits'):
            return url
    
    # If no FITS found, return first URL
    if skymap_urls:
        return skymap_urls[0]
    
    return None


def download(url: str, out_path: Path, timeout: int = 300, 
             chunk_size: int = 8192) -> None:
    """
    Download a file from URL with streaming and timeout handling.
    
    Parameters
    ----------
    url : str
        URL to download
    out_path : Path
        Output file path
    timeout : int, optional
        Request timeout in seconds, by default 300
    chunk_size : int, optional
        Chunk size for streaming download in bytes, by default 8192
    
    Raises
    ------
    requests.RequestException
        If download fails
    """
    # Create parent directory if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True, timeout=timeout)
    response.raise_for_status()
    
    with open(out_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)

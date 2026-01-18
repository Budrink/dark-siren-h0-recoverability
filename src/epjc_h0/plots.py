"""Plotting utilities for H0 depth-drift analysis."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from typing import Optional


def plot_predicted_drift(z: np.ndarray, H0_app_over_H0: np.ndarray, 
                        out_path: Path, title: Optional[str] = None) -> None:
    """
    Plot predicted H0 drift as a function of redshift.
    
    Parameters
    ----------
    z : np.ndarray
        Redshift array
    H0_app_over_H0 : np.ndarray
        Apparent H0 / true H0 ratio
    out_path : Path
        Output file path for the figure
    title : Optional[str], optional
        Plot title. If None, uses default, by default None
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(z, H0_app_over_H0, 'b-', linewidth=2, label=r'$H_0^{\mathrm{app}} / H_0$')
    ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1, alpha=0.5, 
               label='No drift')
    
    ax.set_xlabel('Redshift $z$', fontsize=12)
    ax.set_ylabel(r'$H_0^{\mathrm{app}} / H_0$', fontsize=12)
    ax.set_title(title or 'Predicted H0 Depth-Drift (kappa=1, low-z approximation)', 
                 fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Create output directory if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_skymap_healpix(fits_path: Path, out_path: Path, 
                       title: Optional[str] = None) -> None:
    """
    Plot a HEALPix skymap from a FITS file using Mollweide projection.
    
    Parameters
    ----------
    fits_path : Path
        Path to HEALPix FITS file
    out_path : Path
        Output file path for the figure
    title : Optional[str], optional
        Plot title. If None, uses default, by default None
    """
    # Read HEALPix map
    skymap, header = hp.read_map(str(fits_path), h=True, verbose=False)
    
    # Plot the skymap using healpy's mollview
    # mollview creates its own figure, so we use it directly
    hp.mollview(skymap, title=title or 'GW Skymap', 
                unit='Probability', notext=True, cmap='cylon')
    
    # Get current figure and axes
    fig = plt.gcf()
    ax = plt.gca()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create output directory if needed
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

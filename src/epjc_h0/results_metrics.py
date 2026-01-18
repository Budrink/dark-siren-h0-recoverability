"""Metrics for H0 posterior analysis: probability mass in windows, quantiles, etc."""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from typing import Tuple, Union


# Default windows for CMB and Local H0 estimates
CMB_WINDOW_LO = 65.0  # km/s/Mpc
CMB_WINDOW_HI = 70.0  # km/s/Mpc
LOCAL_WINDOW_LO = 70.0  # km/s/Mpc
LOCAL_WINDOW_HI = 75.0  # km/s/Mpc


def posterior_mass_in_window(h0_grid: np.ndarray, posterior: np.ndarray,
                             h0_lo: float, h0_hi: float) -> float:
    """
    Compute probability mass of posterior in a given H0 window.
    
    Parameters
    ----------
    h0_grid : np.ndarray
        1D array of H0 values (must be monotonic)
    posterior : np.ndarray
        1D array of posterior values (not necessarily normalized)
    h0_lo : float
        Lower bound of H0 window (km/s/Mpc)
    h0_hi : float
        Upper bound of H0 window (km/s/Mpc)
    
    Returns
    -------
    float
        Probability mass in window [h0_lo, h0_hi], in range [0, 1]
    """
    # Normalize posterior
    norm = np.trapz(posterior, h0_grid)
    if norm <= 0:
        return 0.0
    
    posterior_norm = posterior / norm
    
    # Clip window to grid range
    h0_min = h0_grid[0]
    h0_max = h0_grid[-1]
    h0_lo_clipped = max(h0_lo, h0_min)
    h0_hi_clipped = min(h0_hi, h0_max)
    
    # Check if window overlaps with grid
    if h0_lo_clipped >= h0_hi_clipped:
        return 0.0
    
    # Find indices corresponding to window
    idx_lo = np.searchsorted(h0_grid, h0_lo_clipped)
    idx_hi = np.searchsorted(h0_grid, h0_hi_clipped)
    
    # Ensure indices are within bounds
    idx_lo = max(0, idx_lo - 1)  # Include point before if needed
    idx_hi = min(len(h0_grid), idx_hi + 1)  # Include point after if needed
    
    if idx_lo >= idx_hi:
        return 0.0
    
    # Extract window
    h0_window = h0_grid[idx_lo:idx_hi]
    post_window = posterior_norm[idx_lo:idx_hi]
    
    # If window doesn't start/end exactly on grid points, interpolate
    if h0_window[0] < h0_lo_clipped:
        # Interpolate at h0_lo_clipped
        if idx_lo > 0:
            h0_prev = h0_grid[idx_lo - 1]
            post_prev = posterior_norm[idx_lo - 1]
            h0_curr = h0_window[0]
            post_curr = post_window[0]
            if h0_curr != h0_prev:
                t = (h0_lo_clipped - h0_prev) / (h0_curr - h0_prev)
                post_at_lo = post_prev + t * (post_curr - post_prev)
            else:
                post_at_lo = post_curr
            h0_window = np.concatenate([[h0_lo_clipped], h0_window])
            post_window = np.concatenate([[post_at_lo], post_window])
        else:
            h0_window[0] = h0_lo_clipped
    
    if h0_window[-1] > h0_hi_clipped:
        # Interpolate at h0_hi_clipped
        if idx_hi < len(h0_grid):
            h0_curr = h0_window[-1]
            post_curr = post_window[-1]
            h0_next = h0_grid[idx_hi]
            post_next = posterior_norm[idx_hi]
            if h0_next != h0_curr:
                t = (h0_hi_clipped - h0_curr) / (h0_next - h0_curr)
                post_at_hi = post_curr + t * (post_next - post_curr)
            else:
                post_at_hi = post_curr
            h0_window = np.concatenate([h0_window, [h0_hi_clipped]])
            post_window = np.concatenate([post_window, [post_at_hi]])
        else:
            h0_window[-1] = h0_hi_clipped
    
    # Integrate over window
    mass = np.trapz(post_window, h0_window)
    
    return max(0.0, min(1.0, mass))  # Clamp to [0, 1]


def posterior_quantiles(h0_grid: np.ndarray, posterior: np.ndarray,
                        qs: Tuple[float, ...] = (0.16, 0.5, 0.84)) -> np.ndarray:
    """
    Compute quantiles of posterior distribution.
    
    Parameters
    ----------
    h0_grid : np.ndarray
        1D array of H0 values (must be monotonic)
    posterior : np.ndarray
        1D array of posterior values (not necessarily normalized)
    qs : tuple of float, optional
        Quantiles to compute, by default (0.16, 0.5, 0.84)
    
    Returns
    -------
    np.ndarray
        Array of H0 values corresponding to quantiles qs
    """
    # Normalize posterior
    norm = np.trapz(posterior, h0_grid)
    if norm <= 0:
        # Return boundary values if posterior is invalid
        return np.full(len(qs), h0_grid[0] if len(h0_grid) > 0 else 0.0)
    
    posterior_norm = posterior / norm
    
    # Compute CDF using cumulative trapezoid
    cdf = cumulative_trapezoid(posterior_norm, h0_grid, initial=0.0)
    
    # Ensure CDF is normalized (should be, but guard against numerical issues)
    if cdf[-1] > 0:
        cdf = cdf / cdf[-1]
    
    # Find quantiles by interpolation
    quantiles = np.zeros(len(qs))
    for i, q in enumerate(qs):
        # Clamp quantile to [0, 1]
        q_clamped = max(0.0, min(1.0, q))
        
        # Find index where CDF crosses quantile
        idx = np.searchsorted(cdf, q_clamped)
        
        # Handle edge cases
        if idx == 0:
            quantiles[i] = h0_grid[0]
        elif idx >= len(h0_grid):
            quantiles[i] = h0_grid[-1]
        else:
            # Linear interpolation
            cdf_prev = cdf[idx - 1]
            cdf_curr = cdf[idx]
            h0_prev = h0_grid[idx - 1]
            h0_curr = h0_grid[idx]
            
            if cdf_curr != cdf_prev:
                t = (q_clamped - cdf_prev) / (cdf_curr - cdf_prev)
                quantiles[i] = h0_prev + t * (h0_curr - h0_prev)
            else:
                quantiles[i] = h0_prev
    
    return quantiles


def posterior_mass_in_window_centered(h0_grid: np.ndarray, posterior: np.ndarray,
                                      window_center: float, window_halfwidth: float) -> float:
    """
    Compute probability mass of posterior in a window defined by center and half-width.
    
    Parameters
    ----------
    h0_grid : np.ndarray
        1D array of H0 values (must be monotonic)
    posterior : np.ndarray
        1D array of posterior values (not necessarily normalized)
    window_center : float
        Center of H0 window (km/s/Mpc)
    window_halfwidth : float
        Half-width of H0 window (km/s/Mpc)
    
    Returns
    -------
    float
        Probability mass in window [center - halfwidth, center + halfwidth], in range [0, 1]
    """
    h0_lo = window_center - window_halfwidth
    h0_hi = window_center + window_halfwidth
    return posterior_mass_in_window(h0_grid, posterior, h0_lo, h0_hi)

"""
This script generates the guide curve about which the 2D profiles created
using `PoweredNacelleProfile` are revolved.

The nondimensional cross-section coordinates are eta (y) and zeta (z).
"""

__all__ = ["get_guide_curve"]

import sys
import numpy as np
import matplotlib.pyplot as plt


def get_guide_curve(
    N_u, N_l,
    w, h,
    N=100,
):
    
    # From \RCAIDE_LEADS\RCAIDE\Library\Methods\Utilities\Chebyshev\chebyshev_data.py
    eta_range = 0.5 * (1 - np.cos(np.pi * np.arange(0, N) / (N - 1)))  # cosine spaced in range [0,1]

    # Dimensional y-coordinate
    y = (eta_range - 0.5) * w
    
    # Shape function (constant 2 for elliptic lobe)
    S_u = 2.0  # (24)
    S_l = 2.0  # (24)
    
    # Class function
    C_u = eta_range**N_u * (1 - eta_range)**N_u  # (25)
    C_l = eta_range**N_l * (1 - eta_range)**N_l  # (25)
    
    # Nondimensional z-coordinate
    zeta_u = S_u * C_u  # (26)
    zeta_l = S_l * C_l  # (26)
    zeta_u_max = S_u * np.max(C_u)
    zeta_l_max = S_l * np.max(C_l)
    zeta_u_norm = zeta_u / zeta_u_max / 2
    zeta_l_norm = -zeta_l / zeta_l_max / 2
    
    # Dimensional z-coordinate
    z_u = zeta_u_norm * h
    z_l = zeta_l_norm * h
    
    # Non-uniform polar sampling of each half
    r_u_old = np.sqrt(np.array(y)**2 + np.array(z_u)**2)
    theta_u_old = np.rad2deg(np.atan2(np.array(z_u), np.array(y)))
    r_l_old = np.sqrt(np.array(y)**2 + np.array(z_l)**2)
    theta_l_old = np.rad2deg(np.atan2(np.array(z_l), np.array(y)))

    # Resample onto uniform angular grid
    
    # Create uniform theta grid
    theta_grid = np.linspace(0.0, 180.0, N, endpoint=False)
    
    def sample_r_on_theta(theta_old_deg, r_old, theta_query_deg):
        
        # Map to [0,180)
        th_raw = np.array(theta_old_deg, dtype=float)
        th = np.mod(th_raw, 180.0)
        # Keep original 180° as 180, not 0
        mask = (np.isclose(th, 0.0)) & (th_raw > 0)
        th[mask] = 180.0
        
        r = np.array(r_old, dtype=float)
    
        # Compute original Cartesian coordinates
        th_rad = np.deg2rad(th)
        y_old = r * np.cos(th_rad)
        z_old = r * np.sin(th_rad)
    
        # Sort by angle
        order = np.argsort(th)
        ths = th[order]
        ys = y_old[order]
        zs = z_old[order]
    
        # Prepend last-180 and append first+180 for correct periodic wrap
        ths_ext = np.concatenate((ths[-1:] - 180.0, ths, ths[:1] + 180.0))
        ys_ext  = np.concatenate((ys[-1:], ys, ys[:1]))
        zs_ext  = np.concatenate((zs[-1:], zs, zs[:1]))
    
        q = np.mod(theta_query_deg, 180.0).astype(float)
    
        y_query = np.interp(q, ths_ext, ys_ext)
        z_query = np.interp(q, ths_ext, zs_ext)
    
        r_query = np.sqrt(y_query**2 + z_query**2)
        return r_query
    

    # Sample upper and lower radii at the same angular grid
    r_u = sample_r_on_theta(theta_u_old, r_u_old, theta_grid)
    r_l = sample_r_on_theta(theta_l_old, r_l_old, theta_grid)
    
    # Now theta_u and theta_l are the same uniform theta_grid (so different shapes share angles)
    theta_u = theta_grid.copy()
    theta_l = theta_grid.copy()

    return (
        y,
        z_u, z_l,
        r_u, r_l,
        theta_u, theta_l,
    )

#%%

if __name__ == '__main__':
    
    N_u = 0.1
    N_l = 0.1
    w = 1  # 2
    h = 1

    (
    y,
    z_u, z_l,
    r_u, r_l,
    theta_u, theta_l,
    ) = get_guide_curve(
        N_u, N_l,
        w, h,
        N=1000,
    )
    
    
    fig, ax = plt.subplots()
    ax.plot(y, z_u, marker='.')
    ax.plot(y, z_l, marker='.')
    ax.set_aspect('equal')
    plt.show()
    
    
    # Plot r vs theta (now theta is uniform)
    fig, ax = plt.subplots()
    ax.plot(theta_u, r_u/np.min(r_u), marker='.')
    plt.show()
    
    ###
    
    # assume get_guide_curve already defined and returns r_u, r_l, theta_u, theta_l as in your code
    N = 100
    N_u = 0.1
    N_l = 0.1
    w = 1
    h = 1
    y, z_u, z_l, r_u, r_l, theta_u, theta_l = get_guide_curve(N_u, N_l, w, h, N=N)
    
    # 1) Check theta uniformity numerically
    dtheta = np.diff(theta_u)
    print("dtheta min, max, mean:", dtheta.min(), dtheta.max(), dtheta.mean())
    # unwrap (radians) and check differences (robust to wrap)
    theta_rad = np.deg2rad(theta_u)
    dtheta_rad = np.diff(np.unwrap(theta_rad))
    print("dtheta_rad min,max (deg):", np.rad2deg(dtheta_rad).min(), np.rad2deg(dtheta_rad).max())
    
    # 2) Convert resampled polar -> Cartesian
    theta_rad_full = np.deg2rad(theta_u)
    y_from_r = r_u * np.cos(theta_rad_full)
    z_from_r = r_u * np.sin(theta_rad_full)
    
    # 3) Plot original eta-sampled points and theta-resampled points for comparison
    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(y, z_u, '-k', lw=0.5, label='original (eta sampling) curve')
    ax.scatter(y, z_u, c='C0', s=10, label='original sample points')
    ax.scatter(y_from_r, z_from_r, c='C1', s=20, marker='x', label='resampled (uniform theta)')
    ax.set_aspect('equal')
    ax.legend()
    plt.show()
    
    # 4) Optional: show angles between consecutive Cartesian points (perceived spacing)
    # compute Euclidean distances between consecutive Cartesian resampled points
    dists_resampled = np.sqrt(np.sum(np.diff(np.vstack([y_from_r, z_from_r]).T, axis=0)**2, axis=1))
    dists_original = np.sqrt(np.sum(np.diff(np.vstack([y, z_u]).T, axis=0)**2, axis=1))
    print("resampled distances: min,max,mean =", dists_resampled.min(), dists_resampled.max(), dists_resampled.mean())
    print("original distances:  min,max,mean =", dists_original.min(), dists_original.max(), dists_original.mean())
    
    
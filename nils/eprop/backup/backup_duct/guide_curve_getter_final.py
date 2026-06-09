import sys
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["get_guide_curve"]


def get_guide_curve(
    N_u, N_l,
    w, h,
    N=100,
):
    
    # From C:\Users\nmb48\Documents\GitHub\RCAIDE_LEADS\RCAIDE\Library\Methods\Utilities\Chebyshev\chebyshev_data.py
    # cosine spaced in range [0,1]
    eta_range = 0.5*(1 - np.cos(np.pi*np.arange(0,N)/(N-1)))
    
    y = (eta_range - 0.5) * w
    
    zeta_u = 2 * eta_range**N_u * (1 - eta_range)**N_u
    zeta_l = 2 * eta_range**N_l * (1 - eta_range)**N_l
    
    zeta_u_max = 2 * (0.5**N_u) * (0.5**N_u)
    zeta_l_max = 2 * (0.5**N_l) * (0.5**N_l)
    
    zeta_u_norm = zeta_u / zeta_u_max / 2
    zeta_l_norm = -zeta_l / zeta_l_max / 2
    
    z_u = zeta_u_norm * h
    z_l = zeta_l_norm * h
    
    # original polar sampling of each half (non-uniform angles)
    r_u_old = np.sqrt(np.array(y)**2 + np.array(z_u)**2)
    theta_u_old = np.rad2deg(np.atan2(np.array(z_u), np.array(y)))
    
    r_l_old = np.sqrt(np.array(y)**2 + np.array(z_l)**2)
    theta_l_old = np.rad2deg(np.atan2(np.array(z_l), np.array(y)))

    # -------------------- MINIMAL ADDITION: resample onto uniform angular grid --------------------
    # create uniform theta grid (degrees) same for all shapes
    theta_grid = np.linspace(0.0, 180.0, N, endpoint=False)

    # def sample_r_on_theta(theta_old_deg, r_old, theta_query_deg):
    #     """
    #     Periodic interpolation of r(theta). theta_old_deg may be unsorted; map to [0,180).
    #     We sort by theta, append the first point shifted by +180 to allow wrap-around interpolation,
    #     then use numpy.interp to sample at theta_query_deg (assumed in [0,180)).
    #     """
    #     # map to [0,180)
    #     th = np.mod(theta_old_deg, 180.0).astype(float)
    #     r = np.array(r_old, dtype=float)

    #     # sort by angle
    #     order = np.argsort(th)
    #     ths = th[order]
    #     rs = r[order]

    #     # ensure we have at least two points for interpolation
    #     if ths.size < 2:
    #         # degenerate: just repeat
    #         return np.ones_like(theta_query_deg) * (rs[0] if rs.size > 0 else 0.0)

    #     # append first point + 180 for periodic wrap
    #     ths_ext = np.concatenate((ths, ths[:1] + 180.0))
    #     rs_ext = np.concatenate((rs, rs[:1]))

    #     # query: ensure values are within [0,180)
    #     q = np.mod(theta_query_deg, 180.0).astype(float)

    #     # np.interp requires ascending x (ths_ext is ascending)
    #     r_query = np.interp(q, ths_ext, rs_ext)
    #     return r_query
    
    # =============================================================================
    # def sample_r_on_theta(theta_old_deg, r_old, theta_query_deg):
    #     """
    #     Interpolate Cartesian y(θ) and z(θ) (instead of r(θ)) for robustness.
    #     Steps:
    #       - map theta_old to [0,180)
    #       - compute y_old = r_old*cos(theta_old), z_old = r_old*sin(theta_old)
    #       - sort by theta and append first point+180 for wrap
    #       - interpolate y and z separately at theta_query
    #       - return r_query = sqrt(y_q^2 + z_q^2)
    #     """
    #     # map to [0,180)
    #     th = np.mod(theta_old_deg, 180.0).astype(float)
    #     r = np.array(r_old, dtype=float)
    
    #     # compute original Cartesian coordinates
    #     th_rad = np.deg2rad(th)
    #     y_old = r * np.cos(th_rad)
    #     z_old = r * np.sin(th_rad)
    
    #     # sort by angle
    #     order = np.argsort(th)
    #     ths = th[order]
    #     ys = y_old[order]
    #     zs = z_old[order]
    
    #     # ensure we have at least two points for interpolation
    #     if ths.size < 2:
    #         # degenerate: just repeat
    #         if ys.size == 0:
    #             return np.zeros_like(theta_query_deg, dtype=float)
    #         yq = np.ones_like(theta_query_deg) * ys[0]
    #         zq = np.ones_like(theta_query_deg) * zs[0]
    #         return np.sqrt(yq**2 + zq**2)
    
    #     # append first point + 180 for periodic wrap (and corresponding y,z)
    #     ths_ext = np.concatenate((ths, ths[:1] + 180.0))
    #     ys_ext = np.concatenate((ys, ys[:1]))
    #     zs_ext = np.concatenate((zs, zs[:1]))
    
    #     # query: ensure values are within [0,180)
    #     q = np.mod(theta_query_deg, 180.0).astype(float)
    
    #     # interpolate Cartesian components
    #     y_query = np.interp(q, ths_ext, ys_ext)
    #     z_query = np.interp(q, ths_ext, zs_ext)
    
    #     # reconstruct radius
    #     r_query = np.sqrt(y_query**2 + z_query**2)
    #     return r_query
    
    def sample_r_on_theta(theta_old_deg, r_old, theta_query_deg):
        # map to [0,180)
        # th = np.mod(theta_old_deg, 180.0).astype(float)
        # =============================================================================
        # new (minimal)
        th_raw = np.array(theta_old_deg, dtype=float)
        th = np.mod(th_raw, 180.0)          # map into [0,180)
        mask = (np.isclose(th, 0.0)) & (th_raw > 0)
        th[mask] = 180.0                    # keep original 180° as 180, not 0
        # =============================================================================
        r = np.array(r_old, dtype=float)
    
        # compute original Cartesian coordinates
        th_rad = np.deg2rad(th)
        y_old = r * np.cos(th_rad)
        z_old = r * np.sin(th_rad)
    
        # sort by angle
        order = np.argsort(th)
        ths = th[order]
        ys = y_old[order]
        zs = z_old[order]
    
        if ths.size < 2:
            if ys.size == 0:
                return np.zeros_like(theta_query_deg, dtype=float)
            yq = np.ones_like(theta_query_deg) * ys[0]
            zq = np.ones_like(theta_query_deg) * zs[0]
            return np.sqrt(yq**2 + zq**2)
    
        # <-- Minimal fix: prepend last-180 and append first+180 for correct periodic wrap
        ths_ext = np.concatenate((ths[-1:] - 180.0, ths, ths[:1] + 180.0))
        ys_ext  = np.concatenate((ys[-1:],        ys,  ys[:1]))
        zs_ext  = np.concatenate((zs[-1:],        zs,  zs[:1]))
    
        q = np.mod(theta_query_deg, 180.0).astype(float)
    
        y_query = np.interp(q, ths_ext, ys_ext)
        z_query = np.interp(q, ths_ext, zs_ext)
    
        r_query = np.sqrt(y_query**2 + z_query**2)
        return r_query
    # =============================================================================

    # sample upper and lower radii at the same angular grid
    r_u = sample_r_on_theta(theta_u_old, r_u_old, theta_grid)
    r_l = sample_r_on_theta(theta_l_old, r_l_old, theta_grid)
    # -----------------------------------------------------------------------------------------------

    # For backward compatibility keep return signature identical:
    # return (y, z_u, z_l, r_u, r_l, theta_u, theta_l)
    # but now theta_u and theta_l are the same uniform theta_grid (so different shapes share angles)
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
    
    
    fig, ax = plt.subplots()
    # plot r vs theta (now theta is uniform)
    ax.plot(theta_u, r_u/np.min(r_u), marker='.')
    plt.show()
    
    ###
    
    import numpy as np
    import matplotlib.pyplot as plt
    
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
    
    
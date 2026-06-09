"""
Loft a Kulfan-style hollow body and extract slices.

This single script unites two behaviours via the `match_slopes` switch:

- match_slopes=True  : Use the "matched slopes" algorithm (Hermite side_offset
                       that enforces inlet/outlet side slopes while keeping
                       planar inlet/outlet centroids fixed). This is the same
                       logic as in upstream_duct_final_matched_slopes.py.

- match_slopes=False : Use the simpler lofting logic (like upstream_duct_final.py)
                       BUT fix the rotated-section height: the h_inlet/h_outlet
                       values are interpreted as the *post-rotation* (inclined)
                       vertical extent and the code compensates the pre-rotation
                       scaling accordingly.

Notes / caveats:
- All angles passed to the public API are in degrees.
- side_slope_start_deg/end_deg are converted to tangent and used as dz/dx
  endpoint derivatives for a Hermite side_offset (when match_slopes=True).
- The code clamps/falls back to symmetric scaling when requested parameters
  are incompatible (e.g. cos(alpha) ~ 0 or side_offset magnitude too large).
- The function returns a list of closed slice contours: dicts with keys
  's', 'psi_index', 'X', 'Y', 'Z'.
"""

__all__ = ["get_duct_geometry"]

import numpy as np
import matplotlib.pyplot as plt


def hermite_on_x(x, y0, y1, dy0, dy1):
    """
    Evaluate a cubic Hermite interpolant on the domain defined by x.

    x : 1D array of parameter positions (physical coords ok).
    y0, y1 : endpoint values at x[0], x[-1]
    dy0, dy1 : endpoint derivatives dy/dx at x[0], x[-1]

    Returns array same shape as x.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array or sequence")

    x0, x1 = x[0], x[-1]
    if np.isclose(x1, x0):
        raise ValueError("x[0] and x[-1] must be distinct")

    t = (x - x0) / (x1 - x0)
    t = np.clip(t, 0.0, 1.0)
    t2 = t * t
    t3 = t2 * t
    h00 = 2 * t3 - 3 * t2 + 1
    h10 = t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 = t3 - t2

    scale = (x1 - x0)  # dx/dt
    m0 = dy0 * scale
    m1 = dy1 * scale

    y = h00 * y0 + h10 * m0 + h01 * y1 + h11 * m1
    return y


def get_duct_geometry(
    x_centroid_inlet, y_centroid_inlet, z_centroid_inlet,
    x_centroid_outlet, y_centroid_outlet, z_centroid_outlet,
    w_inlet, h_inlet,
    w_outlet, h_outlet,
    angle_inlet, angle_outlet,
    N_inlet,
    N_outlet,
    n_slices,
    include_endpoints=True,
    is_inlet=True,
    # ---- new optional params (minimal addition) ----
    side_slope_start_deg: float = 0.0,
    side_slope_end_deg:   float = 0.0,
    match_slopes: bool = True,
    fig=None,
    ax=None,
):
    """
    Build lofted duct and return closed slices list.

    Parameters
    ----------
    x_centroid_inlet, y_centroid_inlet, z_centroid_inlet : float
        coordinates of inlet planar section centroid (post-rotation).
    x_centroid_outlet, y_centroid_outlet, z_centroid_outlet : float
        coordinates of outlet planar section centroid (post-rotation).
    w_inlet, h_inlet : float
        width and height of inlet cross-section. Height is the post-rotation
        vertical extent (i.e. the value you expect to see after rotating the section).
    w_outlet, h_outlet : float
        width and height of outlet cross-section (post-rotation).
    angle_inlet, angle_outlet : float
        section-plane rotation angles in degrees (rotation about local y-axis).
    N_inlet, N_outlet : float
        Kulfan power-law parameters for template shaping.
    n_slices : int
        number of extracted slices (closed contours).
    include_endpoints : bool
        include exact inlet/outlet slices.
    is_inlet : bool
        influences Hermite slope sign convention for centreline.
    side_slope_start_deg, side_slope_end_deg : float
        angles (deg) specifying desired dz/dx at inlet/outlet for the upper/lower
        side edges. Used only if match_slopes=True.
    match_slopes : bool
        If True, use the matched-slopes algorithm (side_offset Hermite etc).
        If False, use the simpler original lofting logic but correct the
        rotated-section height so h_inlet/h_outlet are post-rotation heights.

    Returns
    -------
    slices_list : list of dicts
        Each dict contains 's', 'psi_index', 'X','Y','Z' arrays forming a closed
        slice contour in physical coords.
    """
    # ------------------ user params (local aliases) ------------------
    w_in = w_inlet
    h_in = h_inlet
    w_out = w_outlet
    h_out = h_outlet

    tilt_end_deg = angle_outlet
    tilt_start_deg = angle_inlet
    N_in = N_inlet
    N_out = N_outlet

    y_center = y_centroid_inlet
    assert y_centroid_inlet == y_centroid_outlet, "y-centroids must match for current impl."
    inlet_z = z_centroid_inlet
    outlet_z = z_centroid_outlet

    L = x_centroid_outlet - x_centroid_inlet
    eta_count = 250
    psi_count = 250

    cubic = lambda ymin, ymax, t: ymin + (ymax - ymin) * (3 * t * t - 2 * t * t * t)

    # create Kulfan-like templates
    eta_range = np.linspace(0.0, 1.0, eta_count)
    psi_range = np.linspace(0.0, 1.0, psi_count)
    eta_grid, psi_grid = np.meshgrid(eta_range, psi_range, indexing='ij')

    x_array = np.zeros_like(eta_grid)
    y_array = np.zeros_like(eta_grid)
    z_pos_array = np.zeros_like(eta_grid)
    z_neg_array = np.zeros_like(eta_grid)

    Nd = 0.005
    eps = 1e-12
    for i_eta, eta in enumerate(eta_range):
        for j_psi, psi in enumerate(psi_range):
            Nc = cubic(N_in, N_out, psi)
            W = 1.0
            H = 1.0
            Sc = 0.5 ** (2 * Nc)
            Cc = eta ** Nc * (1 - eta) ** Nc
            Sd = 0.5 ** (2 * Nd)
            psi_eff = np.clip(psi, eps, 1.0 - eps)
            Cd = psi_eff ** Nd * (1 - psi_eff) ** Nd
            x = x_centroid_inlet + psi * L
            y = -(Sd * Cd) * (1 - 2 * eta) * W / 2.0
            z = Sd * Cd * (Sc * Cc) * H / 2.0
            x_array[i_eta, j_psi] = x
            y_array[i_eta, j_psi] = y
            z_pos_array[i_eta, j_psi] = z
            z_neg_array[i_eta, j_psi] = -z

    col_max_abs_y = np.max(np.abs(y_array), axis=0)
    col_max_abs_y[col_max_abs_y == 0.0] = 1.0
    col_max_zpos = np.max(z_pos_array, axis=0)
    col_max_zpos[col_max_zpos == 0.0] = 1.0

    psi_positions = x_centroid_inlet + psi_range * L
    t_vals = psi_range
    S_vals = cubic(0, 1, t_vals)
    w_cols = w_in + (w_out - w_in) * S_vals
    h_cols = h_in + (h_out - h_in) * S_vals

    # alpha(x) for section-plane rotation
    alpha_cols_rad = np.deg2rad(cubic(tilt_start_deg, tilt_end_deg, t_vals))

    # side_offset Hermite (only meaningful when match_slopes True)
    side_slope_start_tan = np.tan(np.deg2rad(side_slope_start_deg))
    side_slope_end_tan = np.tan(np.deg2rad(side_slope_end_deg))
    side_offset_cols = hermite_on_x(psi_positions, 0.0, 0.0, side_slope_start_tan, side_slope_end_tan)

    # side slope diagnostic
    side_slope_cols_rad = np.deg2rad(cubic(side_slope_start_deg, side_slope_end_deg, t_vals))
    side_slope_cols_tan = np.tan(side_slope_cols_rad)

    # centreline z: choose endpoint slope origin depending on match_slopes
    if match_slopes:
        # use side-slope tangents as centreline endpoint derivatives
        dy0 = side_slope_start_tan
        dy1 = side_slope_end_tan
        z_centerline_cols = hermite_on_x(psi_positions, inlet_z, outlet_z, dy0, dy1)
    else:
        # original behaviour: slopes derived from planar rotation angles
        if is_inlet:
            alpha1_deg, alpha2_deg = angle_inlet, -angle_outlet
        else:
            alpha1_deg, alpha2_deg = -angle_inlet, angle_outlet
        dy0 = np.tan(np.deg2rad(alpha1_deg))
        dy1 = np.tan(np.deg2rad(alpha2_deg))
        # =============================================================================
        z_centerline_cols = hermite_on_x(t_vals, inlet_z, outlet_z, dy0, dy1)
        # z_centerline_cols = hermite_on_x(psi_positions, inlet_z, outlet_z, dy0, dy1)
        # =============================================================================

    # prepare final rotated arrays
    rot_X_pos = np.zeros_like(x_array)
    rot_Y_pos = np.zeros_like(y_array)
    rot_Z_pos = np.zeros_like(z_pos_array)
    rot_X_neg = np.zeros_like(x_array)
    rot_Y_neg = np.zeros_like(y_array)
    rot_Z_neg = np.zeros_like(z_neg_array)

    # per-column loop with two branches (match_slopes True/False)
    small = 1e-12
    clipped_count = 0  # diagnostic if needed

    for j in range(psi_count):
        w_j = w_cols[j]
        h_j = h_cols[j]
        alpha = alpha_cols_rad[j]
        zc = z_centerline_cols[j]
        yc = y_center

        sa = np.sin(alpha)
        ca = np.cos(alpha)

        # lateral scale (same for both modes)
        sy = (w_j / 2.0) / col_max_abs_y[j]

        # template magnitudes
        # =============================================================================
        # m_pos = np.mean(z_pos_array[:, j])
        # m_neg = -np.mean(z_neg_array[:, j])
        m_pos = np.max(z_pos_array[:, j])
        m_neg = -np.min(z_neg_array[:, j])
        # =============================================================================
        m_pos = max(m_pos, 1e-12)
        m_neg = max(m_neg, 1e-12)

        x_col = x_array[:, j]  # array where all entries equal this column's x

        if match_slopes:
            # --- matched-slopes branch: use side_offset_cols + asymmetric scales,
            # and perform the same safeguard/clipping logic as in your matched version.
            side_offset = side_offset_cols[j]

            # SAFEGUARD logic for stability / to prevent inversion:
            if abs(ca) < 1e-6 or ca <= 0.0:
                # rotation near ±90deg or cos negative => cannot satisfy side offset safely
                side_offset = 0.0
                sz_up = (h_j / 2.0) / (max(abs(ca), small) * m_pos)
                sz_dn = (h_j / 2.0) / (max(abs(ca), small) * m_neg)
                clipped_count += 1
            else:
                bound = (h_j / 2.0) / ca
                if bound <= 0.0:
                    side_offset = 0.0
                else:
                    eps_frac = 1e-6
                    side_offset = np.clip(side_offset, - (1.0 - eps_frac) * bound, (1.0 - eps_frac) * bound)
                # solve for asymmetric pre-rotation scales
                sz_up = (h_j / 2.0 / ca - side_offset) / m_pos
                sz_dn = (h_j / 2.0 / ca + side_offset) / m_neg
                # numeric guard: fall back to symmetric if insane
                max_scale = 1e6
                if not (0.0 < sz_up < max_scale and 0.0 < sz_dn < max_scale):
                    side_offset = 0.0
                    sz_up = (h_j / 2.0) / (max(abs(ca), small) * m_pos)
                    sz_dn = (h_j / 2.0) / (max(abs(ca), small) * m_neg)
                    clipped_count += 1

            # scale templates and add axial offset
            y_col = y_array[:, j] * sy
            zpos_col = z_pos_array[:, j] * sz_up + side_offset
            zneg_col = z_neg_array[:, j] * sz_dn + side_offset
            
            # =============================================================================
            z_all = np.concatenate((zpos_col, zneg_col))
            z_mean = np.mean(z_all)
            
            zpos_col -= z_mean
            zneg_col -= z_mean
            # =============================================================================

            # rotate
            dx_pos = sa * zpos_col
            zpos_r = ca * zpos_col
            Xpos = x_col + dx_pos
            Ypos = yc + y_col
            Zpos = zc + zpos_r

            dx_neg = sa * zneg_col
            zneg_r = ca * zneg_col
            Xneg = x_col + dx_neg
            Yneg = yc + y_col
            Zneg = zc + zneg_r

        else:
            # --- simple branch: do not attempt side_offset or asymmetric scaling ---
            # But correct the pre-rotation vertical scale so that *after rotation*
            # the post-rotation half-height equals h_j/2 (user expects h_j to be post-rotation).
            # sz_pre = (h_j/2) / (col_max_zpos[j] * cos(alpha))
            # Use abs(ca) because height magnitude should be positive; guard by small.
            ca_mag = max(abs(ca), small)
            sz = (h_j / 2.0) / (col_max_zpos[j] * ca_mag)

            # scale templates (symmetric)
            y_col = y_array[:, j] * sy
            zpos_col = z_pos_array[:, j] * sz
            zneg_col = z_neg_array[:, j] * sz
            
            # =============================================================================
            z_all = np.concatenate((zpos_col, zneg_col))
            z_mean = np.mean(z_all)
            
            zpos_col -= z_mean
            zneg_col -= z_mean
            # =============================================================================

            # rotate about local y
            dx_pos = sa * zpos_col
            zpos_r = ca * zpos_col
            Xpos = x_col + dx_pos
            Ypos = yc + y_col
            Zpos = zc + zpos_r

            dx_neg = sa * zneg_col
            zneg_r = ca * zneg_col
            Xneg = x_col + dx_neg
            Yneg = yc + y_col
            Zneg = zc + zneg_r

        rot_X_pos[:, j] = Xpos
        rot_Y_pos[:, j] = Ypos
        rot_Z_pos[:, j] = Zpos
        rot_X_neg[:, j] = Xneg
        rot_Y_neg[:, j] = Yneg
        rot_Z_neg[:, j] = Zneg

    # optional: you could return or print clipped_count for diagnostics
    if clipped_count:
        # do not spam the user — short message only
        print(f"get_duct_geometry: applied {clipped_count} fallback/clips to keep geometry valid.")

    # ---------------------------------------------------------------------
    # Extract closed slices at requested parametric s positions
    if include_endpoints:
        s_list = np.linspace(0.0, 1.0, num=n_slices)
    else:
        s_list = np.linspace(0.0, 1.0, num=n_slices + 2)[1:-1]

    j_indices = np.clip(np.round(s_list * (psi_count - 1)).astype(int), 0, psi_count - 1)

    slices_list = []
    for s_val, j in zip(s_list, j_indices):
        Xu = rot_X_pos[:, j].copy()
        Yu = rot_Y_pos[:, j].copy()
        Zu = rot_Z_pos[:, j].copy()
        Xl = rot_X_neg[:, j].copy()
        Yl = rot_Y_neg[:, j].copy()
        Zl = rot_Z_neg[:, j].copy()

        X_closed = np.concatenate((Xu, Xl[::-1]))
        Y_closed = np.concatenate((Yu, Yl[::-1]))
        Z_closed = np.concatenate((Zu, Zl[::-1]))

        slices_list.append({
            's': float(s_val),
            'psi_index': int(j),
            'X': X_closed,
            'Y': Y_closed,
            'Z': Z_closed
        })

    # Plot result (kept in function for convenience)
    if (fig == None and ax == None):
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(rot_X_pos, rot_Y_pos, rot_Z_pos,
                    rstride=3, cstride=3, linewidth=0.2, edgecolor='gray',
                    facecolor='lightcyan', alpha=0.2)

    ax.plot_surface(rot_X_neg, rot_Y_neg, rot_Z_neg,
                    rstride=3, cstride=3, linewidth=0.2, edgecolor='gray',
                    facecolor='lightcyan', alpha=0.2)

    for sl in slices_list:
        ax.plot(sl['X'], sl['Y'], sl['Z'], color='k', linewidth=1.6)

    ax.plot(psi_positions, np.ones_like(psi_positions) * y_center, z_centerline_cols,
            color='k', linewidth=2, label='centreline')

    # if (fig == None and ax == None):

    ax.set_xlabel('X (axial)')
    ax.set_ylabel('Y (lateral)')
    ax.set_zlabel('Z (vertical)')
    mode = "matched_slopes" if match_slopes else "simple_corrected_height"
    ax.set_title(f'Loft ({mode}): (w_in,h_in) -> (w_out,h_out) with tilt α(x) and centreline')
    ax.view_init(elev=20, azim=120)
    try:
        ax.set_box_aspect((np.ptp(rot_X_pos), np.ptp(rot_Y_pos),
                           np.ptp(np.concatenate((rot_Z_pos.ravel(), rot_Z_neg.ravel())))))
    except Exception:
        pass
        
    if (fig == None and ax == None):
        plt.tight_layout()
        plt.show()

    return slices_list


# ----------------------------
# Diagnostic plotting in main
# ----------------------------
if __name__ == "__main__":
    # example parameters (adapted from your block)
    dz_hx_intake = -0.5
    w_intake = 0.75
    h_intake = 0.25
    l_up_duct = 3
    l_hx = 1.5
    w_hx = 1.0
    h_hx = 0.75
    dx_hx_corner = 0.5
    dz_hx_corner = 0.15
    l_down_duct = 2
    dz_hx_fan = 0.5
    d_fan = 0.5
    l_fan = 0.3
    d_nozzle = 0.3
    l_nozzle = 0.3

    x_centroid_hx = 0.0
    y_centroid_hx = 0.0
    z_centroid_hx = 0.0

    # choose one of the two start/end configurations you used earlier
    x_centroid_inlet = x_centroid_hx - l_hx / 2 - l_up_duct
    y_centroid_inlet = y_centroid_hx
    z_centroid_inlet = z_centroid_hx + dz_hx_intake
    x_centroid_outlet = x_centroid_hx - (l_hx / 2 - dx_hx_corner) / 2
    y_centroid_outlet = y_centroid_hx
    z_centroid_outlet = z_centroid_hx - (h_hx / 2 - dz_hx_corner) / 2
    w_inlet = w_intake
    h_inlet = h_intake
    w_outlet = w_hx
    # Choose h_outlet as the *post-rotation* height you want to see.
    # The simple branch will correct for rotation; matched branch expects the same.
    h_outlet = (h_hx / 2 + dz_hx_corner)
    angle_inlet = 0.0
    angle_outlet = -(90 - 22.78)
    N_inlet = 0.25
    N_outlet = 0.005
    n_slices = 10
    include_endpoints = True
    is_inlet = True

    side_start_deg = 0.0
    side_end_deg = 10.0

    # choose mode
    match_slopes_flag = False   # set to False to run the simple corrected-height branch
    
    # =============================================================================
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    for match_slopes_flag in [False, True]:
    # =============================================================================

        slices = get_duct_geometry(
            x_centroid_inlet, y_centroid_inlet, z_centroid_inlet,
            x_centroid_outlet, y_centroid_outlet, z_centroid_outlet,
            w_inlet, h_inlet,
            w_outlet, h_outlet,
            angle_inlet, angle_outlet,
            N_inlet, N_outlet,
            n_slices, include_endpoints, is_inlet,
            side_slope_start_deg=side_start_deg,
            side_slope_end_deg=side_end_deg,
            match_slopes=match_slopes_flag,
            # =============================================================================
            fig=fig,
            ax=ax,
            # =============================================================================
        )
        
        # =============================================================================
        ax.set_xlabel('X (axial)')
        ax.set_ylabel('Y (lateral)')
        ax.set_zlabel('Z (vertical)')
        mode = "matched_slopes" if match_slopes_flag else "simple_corrected_height"
        ax.set_title(f'Loft ({mode}): (w_in,h_in) -> (w_out,h_out) with tilt α(x) and centreline')
        ax.view_init(elev=20, azim=120)
        # try:
        #     ax.set_box_aspect((np.ptp(rot_X_pos), np.ptp(rot_Y_pos),
        #                        np.ptp(np.concatenate((rot_Z_pos.ravel(), rot_Z_neg.ravel())))))
        # except Exception:
        #     pass
        plt.tight_layout()
        plt.show()
        
    # import sys
    # sys.exit()
    # =============================================================================

    # Recompute 1D diagnostics for plotting
    L = x_centroid_outlet - x_centroid_inlet
    psi_count = 250
    psi_range = np.linspace(0.0, 1.0, psi_count)
    psi_positions = x_centroid_inlet + psi_range * L
    cubic = lambda ymin, ymax, t: ymin + (ymax - ymin) * (3 * t * t - 2 * t * t * t)
    t_vals = psi_range
    S_vals = cubic(0, 1, t_vals)
    w_cols = w_inlet + (w_outlet - w_inlet) * S_vals
    h_cols = h_inlet + (h_outlet - h_inlet) * S_vals
    alpha_cols_rad = np.deg2rad(cubic(angle_inlet, angle_outlet, t_vals))
    alpha_cols_deg = np.rad2deg(alpha_cols_rad)
    side_slope_cols_rad = np.deg2rad(cubic(side_start_deg, side_end_deg, t_vals))
    side_slope_cols_deg = np.rad2deg(side_slope_cols_rad)
    side_offset_cols = hermite_on_x(psi_positions, 0.0, 0.0,
                                    np.tan(np.deg2rad(side_start_deg)),
                                    np.tan(np.deg2rad(side_end_deg)))
    if match_slopes_flag:
        z_centerline_cols = hermite_on_x(psi_positions, z_centroid_inlet, z_centroid_outlet,
                                         np.tan(np.deg2rad(side_start_deg)),
                                         np.tan(np.deg2rad(side_end_deg)))
    else:
        if is_inlet:
            alpha1_deg, alpha2_deg = angle_inlet, -angle_outlet
        else:
            alpha1_deg, alpha2_deg = -angle_inlet, angle_outlet
            # =============================================================================
            z_centerline_cols = hermite_on_x(t_vals, z_centroid_inlet, z_centroid_outlet,
                                             np.tan(np.deg2rad(alpha1_deg)),
                                             np.tan(np.deg2rad(alpha2_deg)))
            # z_centerline_cols = hermite_on_x(psi_positions, z_centroid_inlet, z_centroid_outlet,
            #                                  np.tan(np.deg2rad(alpha1_deg)),
            #                                  np.tan(np.deg2rad(alpha2_deg)))
            # =============================================================================

    # diagnostics plotting
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes = axes.ravel()
    axes[0].plot(psi_positions, w_cols, label='w(x)')
    axes[0].set_ylabel('width (m)'); axes[0].grid(True); axes[0].legend()
    axes[1].plot(psi_positions, h_cols, label='h(x)'); axes[1].set_ylabel('height (m)'); axes[1].grid(True); axes[1].legend()
    axes[2].plot(psi_positions, alpha_cols_deg, label='alpha(x) [deg]'); axes[2].set_ylabel('α (deg)'); axes[2].grid(True); axes[2].legend()
    axes[3].plot(psi_positions, side_slope_cols_deg, label='side_slope (deg)'); axes[3].set_ylabel('side slope (deg)'); axes[3].grid(True); axes[3].legend()
    axes[4].plot(psi_positions, side_offset_cols, label='side_offset (m) (Hermite)'); axes[4].set_ylabel('side offset (m)'); axes[4].grid(True); axes[4].legend()
    axes[5].plot(psi_positions, z_centerline_cols, label='z_centerline (m)'); axes[5].set_ylabel('z_centerline (m)'); axes[5].set_xlabel('X (axial)'); axes[5].grid(True); axes[5].legend()
    plt.suptitle(f'Cubic/Hermite profiles along centreline (match_slopes={match_slopes_flag})')
    plt.tight_layout(rect=[0, 0.0, 1, 0.97])
    plt.show()
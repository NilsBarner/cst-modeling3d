"""
Loft a Kulfan-style hollow body:
 - separate cubic interpolation of width (w) and height (h) from inlet -> outlet
 - cubic tilt alpha(x) about local y-axis
 - loft along a curved centreline z_c(x) (constant y_center)
Fully self-contained.  Adjust user params in the "User parameters" block.
"""

__all__ = ["get_duct_geometry"]

import numpy as np
import matplotlib.pyplot as plt

def hermite_on_x(x, y0, y1, dy0, dy1):
    """
    Evaluate a cubic Hermite interpolant on the domain defined by x.

    Parameters
    - x : array_like
        1D array of parameter positions (can be floats, not necessarily in [0,1]).
        The function uses x[0] and x[-1] as the interpolation endpoints.
    - y0 : float
        Value at x[0].
    - y1 : float
        Value at x[-1].
    - dy0 : float
        Derivative dy/dx at x[0].
    - dy1 : float
        Derivative dy/dx at x[-1].

    Returns
    - y : ndarray
        Array of the same shape as x with the Hermite cubic values.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array or sequence")

    x0, x1 = x[0], x[-1]
    if np.isclose(x1, x0):
        raise ValueError("x[0] and x[-1] must be distinct")

    # map x -> t in [0,1]
    t = (x - x0) / (x1 - x0)
    # clamp t to [0,1] to avoid extrapolation artifacts
    t = np.clip(t, 0.0, 1.0)

    # Hermite basis functions
    t2 = t * t
    t3 = t2 * t
    h00 = 2*t3 - 3*t2 + 1
    h10 = t3 - 2*t2 + t
    h01 = -2*t3 + 3*t2
    h11 = t3 - t2

    # convert endpoint derivatives dy/dx -> dy/dt
    scale = (x1 - x0)            # dx/dt
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
):
    """
    (same docstring as before) + optional side tilt params:
    side_up_start_deg, side_up_end_deg:
        angle (degrees) that the *upper* side edge makes at inlet/outlet (cubic blend).
    side_dn_start_deg, side_dn_end_deg:
        angle (degrees) that the *lower* side edge makes at inlet/outlet (cubic blend).

    The side angles are added to the section plane rotation angle alpha(x) to form alpha_up, alpha_dn.
    """
    
    # ------------------ User parameters ------------------
    # Cross-section dimension endpoints (measured in a vertical x=const plane)
    w_in = w_inlet  # inlet width (y-direction total)
    h_in = h_inlet  # inlet height (z-direction total)
    
    w_out = w_outlet  # outlet width
    h_out = h_outlet  # outlet height
    
    # Tilt (rotation about local y axis)
    tilt_end_deg = angle_outlet  # final tilt angle at outlet (degrees). inlet tilt assumed 0.
    tilt_start_deg = angle_inlet
    N_in = N_inlet
    N_out = N_outlet
    
    # Centreline / loft path
    y_center = y_centroid_inlet
    assert y_centroid_inlet == y_centroid_outlet
    inlet_z = z_centroid_inlet  # 0.0      # z of inlet centroid (user-specified)
    outlet_z = z_centroid_outlet  # 0.4     # z of outlet centroid
    
    # geometry and sampling
    L = x_centroid_outlet - x_centroid_inlet
    eta_count = 250
    psi_count = 250
    
    # smoothing toggles (use cubic smoothstep)
    use_smooth = True
    
    # --- NEW: slice extraction params (minimal addition) ---
    n_slices = n_slices  # 3                 # how many slices to extract (closed contours)
    include_endpoints = include_endpoints  # True     # include inlet (psi=0) and outlet (psi=1) if True

    # keep unchanged parts up to computing S_vals, w_cols, h_cols, alpha_cols_rad ...
    cubic = lambda ymin, ymax, t: ymin + (ymax - ymin) * (3*t*t - 2*t*t*t)

    # grids
    eta_range = np.linspace(0.0, 1.0, eta_count)
    psi_range = np.linspace(0.0, 1.0, psi_count)
    eta_grid, psi_grid = np.meshgrid(eta_range, psi_range, indexing='ij')

    # intermediate arrays
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
            Sc = 0.5**(2 * Nc)
            Cc = eta**Nc * (1 - eta)**Nc
            Sd = 0.5**(2 * Nd)
            psi_eff = np.clip(psi, eps, 1.0 - eps)
            Cd = psi_eff**Nd * (1 - psi_eff)**Nd
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
    
    # =============================================================================
    # --- NEW: side_offset profile (Hermite) so top/bottom edges have specified endpoint slopes
    # use physical x positions for Hermite interpolation
    psi_positions = x_centroid_inlet + psi_range * L
    side_slope_start_tan = np.tan(np.deg2rad(side_slope_start_deg))
    side_slope_end_tan   = np.tan(np.deg2rad(side_slope_end_deg))
    # side_offset is zero at endpoints but has derivative = requested slope (dz/dx) at endpoints
    side_offset_cols = hermite_on_x(psi_positions, 0.0, 0.0, side_slope_start_tan, side_slope_end_tan)
    # =============================================================================

    # section plane rotation (alpha)
    alpha_cols_rad = np.deg2rad(cubic(tilt_start_deg, tilt_end_deg, t_vals))

    # --- NEW: side-tilt cubic profiles (upper and lower) ---
    # single side-slope (same for top and bottom) blended cubically from inlet->outlet
    side_slope_cols_rad = np.deg2rad(cubic(side_slope_start_deg, side_slope_end_deg, t_vals))
    side_slope_cols_tan = np.tan(side_slope_cols_rad)   # dz/dx for each column

    # centreline z (unchanged uses hermite)
    if is_inlet == True:
        alpha1_deg, alpha2_deg = angle_inlet, -angle_outlet
    elif is_inlet == False:
        alpha1_deg, alpha2_deg = -angle_inlet, angle_outlet
    dy0 = np.tan(np.deg2rad(alpha1_deg))
    dy1 = np.tan(np.deg2rad(alpha2_deg))
    z_centerline_cols = hermite_on_x(t_vals, inlet_z, outlet_z, dy0, dy1)
    
    # =============================================================================
    # centreline z now uses side_slope tangents so centreline slope matches side slopes
    dy0 = side_slope_start_tan
    dy1 = side_slope_end_tan
    z_centerline_cols = hermite_on_x(psi_positions, inlet_z, outlet_z, dy0, dy1)
    # =============================================================================

    # prepare arrays
    rot_X_pos = np.zeros_like(x_array)
    rot_Y_pos = np.zeros_like(y_array)
    rot_Z_pos = np.zeros_like(z_pos_array)
    rot_X_neg = np.zeros_like(x_array)
    rot_Y_neg = np.zeros_like(y_array)
    rot_Z_neg = np.zeros_like(z_neg_array)

    small = 1e-8  # safeguard to avoid division by zero on cos()

    # Loop columns
    for j in range(psi_count):
        w_j = w_cols[j]
        h_j = h_cols[j]
        alpha = alpha_cols_rad[j]
        zc = z_centerline_cols[j]
        yc = y_center
        
        # =============================================================================
        # # lateral scale (unchanged)
        # sy = (w_j / 2.0) / col_max_abs_y[j]
        
        # # rotation trig
        # sa = np.sin(alpha)
        # ca = np.cos(alpha)
        # small = 1e-12
        
        # # magnitudes representing the template vertical shape for column j
        # m_pos = np.mean(z_pos_array[:, j])         # positive mean for upper template
        # m_neg = -np.mean(z_neg_array[:, j])       # positive magnitude for lower template
        
        # # base axial for column (same for all eta) — physical x position
        # x_col = x_array[:, j]                     # array filled with same x value
        
        # # side_offset for this column (Hermite curve, zero at endpoints, slope set by user)
        # side_offset = side_offset_cols[j]         # scalar
        
        # # Solve for asymmetric pre-rotation vertical scales so that after rotation:
        # #   ca*( sz_up*m_pos + side_offset ) ==  h_j/2
        # #   ca*( -sz_dn*m_neg + side_offset ) == -h_j/2
        # # => sz_up = ( h_j/(2*ca) - side_offset ) / m_pos
        # #    sz_dn = ( h_j/(2*ca) + side_offset ) / m_neg
        # # Guard against ca ~= 0 or tiny m_pos/m_neg
        # # if abs(ca) < small or abs(m_pos) < small or abs(m_neg) < small:
        # #     # fallback symmetric scaling (safe numeric fallback)
        # #     sz_up = (h_j / 2.0) / (max(abs(ca), small) * max(m_pos, small))
        # #     sz_dn = (h_j / 2.0) / (max(abs(ca), small) * max(m_neg, small))
        # # else:
        # sz_up = (h_j / 2.0 / ca - side_offset) / m_pos
        # sz_dn = (h_j / 2.0 / ca + side_offset) / m_neg
        
        # # scale templates and add side_offset (we add side_offset here as that is the
        # # axial-dependent z-shift that enforces the specified slopes at the ends)
        # y_col = y_array[:, j] * sy
        # zpos_col = z_pos_array[:, j] * sz_up + side_offset   # upper pre-rotation offsets
        # zneg_col = z_neg_array[:, j] * sz_dn + side_offset   # lower pre-rotation offsets (negative values)
        
        # # rotate about local y (single rotation for entire section)
        # dx_pos = sa * zpos_col
        # zpos_r = ca * zpos_col
        # Xpos = x_col + dx_pos
        # Ypos = yc + y_col
        # Zpos = zc + zpos_r
        
        # dx_neg = sa * zneg_col
        # zneg_r = ca * zneg_col
        # Xneg = x_col + dx_neg
        # Yneg = yc + y_col
        # Zneg = zc + zneg_r
        
        
        # lateral scale (unchanged)
        sy = (w_j / 2.0) / col_max_abs_y[j]
        
        # rotation trig
        sa = np.sin(alpha)
        ca = np.cos(alpha)
        small = 1e-12
        
        # columnwise template magnitudes (positive)
        m_pos = np.mean(z_pos_array[:, j])         # positive mean for upper template
        m_neg = -np.mean(z_neg_array[:, j])        # positive magnitude for lower template
        m_pos = max(m_pos, 1e-12)
        m_neg = max(m_neg, 1e-12)
        
        # physical x for this column (all entries equal)
        x_col = x_array[:, j]
        
        # requested axial side offset at this column (Hermite curve; zero at endpoints)
        side_offset = side_offset_cols[j]
        
        # SAFEGUARD 1: if cos(alpha) is tiny or negative (ill-conditioned) fallback
        # (rotation near ±90deg cannot reliably satisfy both heights)
        if abs(ca) < 1e-6 or ca <= 0.0:
            # graceful fallback: disable side offset effect for this column,
            # use symmetric pre-rotation vertical scale (safe and stable).
            side_offset = 0.0
            sz_up = (h_j / 2.0) / (max(abs(ca), small) * m_pos)
            sz_dn = (h_j / 2.0) / (max(abs(ca), small) * m_neg)
        else:
            # SAFEGUARD 2: clamp side_offset so numerators for sz_up/sz_dn remain positive
            # numerators must satisfy:
            #   h/(2*ca) - side_offset > 0   and   h/(2*ca) + side_offset > 0
            # => -h/(2*ca) < side_offset < h/(2*ca)
            bound = (h_j / 2.0) / ca
            # If bound is negative (shouldn't happen because we required ca>0 above),
            # treat as fallback (should not reach here).
            if bound <= 0.0:
                side_offset = 0.0
            else:
                # keep a small safety margin; prevents hitting zero numerators exactly
                eps_frac = 1e-6
                side_offset = np.clip(side_offset, - (1.0 - eps_frac) * bound, (1.0 - eps_frac) * bound)
        
            # now compute scales (guaranteed positive numerators)
            sz_up = (h_j / 2.0 / ca - side_offset) / m_pos
            sz_dn = (h_j / 2.0 / ca + side_offset) / m_neg
        
            # additional numeric guard: if any scale gets absurd, fall back to symmetric
            max_scale = 1e6
            if not (0.0 < sz_up < max_scale and 0.0 < sz_dn < max_scale):
                # fallback symmetric scaling (stable)
                side_offset = 0.0
                sz_up = (h_j / 2.0) / (max(abs(ca), small) * m_pos)
                sz_dn = (h_j / 2.0) / (max(abs(ca), small) * m_neg)
        
        # scale templates and add side_offset (applied equally to top and bottom pre-rotation)
        y_col = y_array[:, j] * sy
        zpos_col = z_pos_array[:, j] * sz_up + side_offset   # upper pre-rotation offsets
        zneg_col = z_neg_array[:, j] * sz_dn + side_offset   # lower pre-rotation offsets (negative values)
        
        # rotate about local y (single rotation for entire section)
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
        # =============================================================================

        rot_X_pos[:, j] = Xpos
        rot_Y_pos[:, j] = Ypos
        rot_Z_pos[:, j] = Zpos
        rot_X_neg[:, j] = Xneg
        rot_Y_neg[:, j] = Yneg
        rot_Z_neg[:, j] = Zneg

    # (the rest of the function — slice extraction + plotting + return — is unchanged)
    # ----------------- (copy over the slice extraction + plotting + return from your original)
    # Decide parametric s positions for slices along centreline.
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

    # plotting / prints kept identical to original
    
    # Plot result
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # plot upper and lower surfaces
    ax.plot_surface(rot_X_pos, rot_Y_pos, rot_Z_pos,
                    rstride=3, cstride=3, linewidth=0.2, edgecolor='gray',
                    facecolor='lightcyan', alpha=0.2)
    
    ax.plot_surface(rot_X_neg, rot_Y_neg, rot_Z_neg,
                    rstride=3, cstride=3, linewidth=0.2, edgecolor='gray',
                    facecolor='lightcyan', alpha=0.2)
    
    # overlay extracted slice contours onto the surface (bold black lines)
    for sl in slices_list:
        ax.plot(sl['X'], sl['Y'], sl['Z'], color='k', linewidth=1.6)
    
    # plot centreline
    xc_line = psi_positions
    zc_line = z_centerline_cols
    yc_line = np.ones_like(xc_line) * y_center
    ax.plot(xc_line, yc_line, zc_line, color='k', linewidth=2, label='centreline')
    
    ax.set_xlabel('X (axial)')
    ax.set_ylabel('Y (lateral)')
    ax.set_zlabel('Z (vertical)')
    ax.set_title('Loft between (w_in,h_in) -> (w_out,h_out) with tilt α(x) and curved centreline')
    
    # set view and aspect as reasonable
    ax.view_init(elev=20, azim=120)
    try:
        ax.set_box_aspect((np.ptp(rot_X_pos), np.ptp(rot_Y_pos), np.ptp(np.concatenate((rot_Z_pos.ravel(), rot_Z_neg.ravel())))))
    except Exception:
        pass
    
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return slices_list

#%%

if __name__ == '__main__':
    
    # x_centroid_inlet = 0.0
    # y_centroid_inlet = 0.0
    # z_centroid_inlet = 0.0
    # x_centroid_outlet = 2.0
    # y_centroid_outlet = 0.0
    # z_centroid_outlet = 0.4
    # w_inlet = 0.6
    # h_inlet = 0.6
    # w_outlet = 1.0
    # h_outlet = 0.6
    # angle_inlet = 45.0
    # angle_outlet = 25.0
    # N_inlet = 0.25
    # N_outlet = 0.005
    # n_slices = 3
    # is_inlet = True
    
    ###
    
    # Intake
    dz_hx_intake = -0.5
    w_intake = 0.75
    h_intake = 0.25
    # Upstream duct
    l_up_duct = 3  # 1
    # HX
    l_hx = 1.5
    w_hx = 1
    h_hx = 0.75
    dx_hx_corner = 0.5
    dz_hx_corner = 0.15
    # Downstream duct
    l_down_duct = 2  # 0.5
    # Fan
    dz_hx_fan = 0.5
    d_fan = 0.5
    l_fan = 0.3
    # Nozzle
    d_nozzle = 0.3
    l_nozzle = 0.3
    
    x_centroid_hx = 0.0
    y_centroid_hx = 0.0
    z_centroid_hx = 0.0
    
    ###
    
    x_centroid_inlet = x_centroid_hx - l_hx/2 - l_up_duct
    y_centroid_inlet = y_centroid_hx
    z_centroid_inlet = z_centroid_hx + dz_hx_intake
    x_centroid_outlet = x_centroid_hx - (l_hx/2 - dx_hx_corner) / 2
    y_centroid_outlet = y_centroid_hx
    z_centroid_outlet = z_centroid_hx - (h_hx/2 - dz_hx_corner) / 2
    w_inlet = w_intake
    h_inlet = h_intake
    w_outlet = w_hx
    h_outlet = (h_hx/2 + dz_hx_corner)# / np.cos(np.deg2rad(90 - 22.78))
    angle_inlet = 0.0
    # =============================================================================
    angle_outlet = -(90 - 22.78)
    alpha = 22.78
    beta = 115.23
    # angle_outlet = -(180 - (alpha + beta))
    # =============================================================================
    N_inlet = 0.25
    N_outlet = 0.005
    n_slices = 10
    include_endpoints = True
    is_inlet = True
    
    ###
    
    # x_centroid_inlet = x_centroid_hx + (l_hx/2 - dx_hx_corner) / 2
    # y_centroid_inlet = y_centroid_hx
    # z_centroid_inlet = z_centroid_hx + (h_hx/2 - dz_hx_corner) / 2
    # x_centroid_outlet = x_centroid_hx + l_hx/2 + l_down_duct
    # y_centroid_outlet = y_centroid_hx
    # z_centroid_outlet = z_centroid_hx + dz_hx_fan
    # w_inlet = w_hx
    # h_inlet = (h_hx/2 + dz_hx_corner)# / np.cos(np.deg2rad(90 - 22.78))
    # w_outlet = d_fan
    # h_outlet = d_fan
    # angle_inlet = -(90 - 22.78)
    # angle_outlet = 0.0
    # N_inlet = 0.005
    # N_outlet = 0.5
    # n_slices = 10
    # include_endpoints = False
    # is_inlet = False
    
    # =============================================================================
    #     slices_list = get_duct_geometry(
    #         x_centroid_inlet, y_centroid_inlet, z_centroid_inlet,
    #         x_centroid_outlet, y_centroid_outlet, z_centroid_outlet,
    #         w_inlet, h_inlet,
    #         w_outlet, h_outlet,
    #         angle_inlet, angle_outlet,
    #         N_inlet,
    #         N_outlet,
    #         n_slices,
    #         include_endpoints,
    #         is_inlet,
    #     )
    
    slices = get_duct_geometry(
        x_centroid_inlet, y_centroid_inlet, z_centroid_inlet,
        x_centroid_outlet, y_centroid_outlet, z_centroid_outlet,
        w_inlet, h_inlet,
        w_outlet, h_outlet,
        angle_inlet, angle_outlet,
        N_inlet, N_outlet,
        n_slices, include_endpoints, is_inlet,
        # side_slope_start_deg=0.0,
        # side_slope_end_deg=0.0,
        # =============================================================================
        side_slope_start_deg=0.0,
        side_slope_end_deg=180 - (alpha + beta),
        # =============================================================================
    )
    # =============================================================================






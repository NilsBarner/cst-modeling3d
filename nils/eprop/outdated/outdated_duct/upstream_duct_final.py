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
):

    # ------------------ User parameters ------------------
    # Cross-section dimension endpoints (measured in a vertical x=const plane)
    w_in = w_inlet  # 0.6        # inlet width (y-direction total)
    h_in = h_inlet  # 0.6        # inlet height (z-direction total)
    
    w_out = w_outlet  # 1.0       # outlet width
    h_out = h_outlet  # 0.6       # outlet height
    
    # Tilt (rotation about local y axis)
    tilt_end_deg = angle_outlet  # 25.0    # final tilt angle at outlet (degrees). inlet tilt assumed 0.
    # =============================================================================
    tilt_start_deg = angle_inlet  # 45.0
    N_in = N_inlet  # 0.25
    N_out = N_outlet  # 0.005
    # =============================================================================
    
    # Centreline / loft path
    # =============================================================================
    # y_center = 0.0     # constant y of centreline
    y_center = y_centroid_inlet
    assert y_centroid_inlet == y_centroid_outlet
    # =============================================================================
    inlet_z = z_centroid_inlet  # 0.0      # z of inlet centroid (user-specified)
    outlet_z = z_centroid_outlet  # 0.4     # z of outlet centroid
    
    # geometry and sampling
    # =============================================================================
    # L = 2.0            # axial length
    L = x_centroid_outlet - x_centroid_inlet
    # =============================================================================
    # =============================================================================
    # eta_count = 120
    # psi_count = 120
    eta_count = 250
    psi_count = 250
    # =============================================================================
    
    # smoothing toggles (use cubic smoothstep)
    use_smooth = True
    
    # --- NEW: slice extraction params (minimal addition) ---
    n_slices = n_slices  # 3                 # how many slices to extract (closed contours)
    include_endpoints = include_endpoints  # True     # include inlet (psi=0) and outlet (psi=1) if True
    # -------------------------------------------------------
    
    # -----------------------------------------------------
    
    # Kulfan-esque cubic used in your script
    # =============================================================================
    # cubic = lambda x: 0.5 * (1.001 + 2 * x**3 - 3 * x**2)
    cubic = lambda ymin, ymax, t: ymin + (ymax - ymin) * (3*t*t - 2*t*t*t)
    # =============================================================================
    # Smoothstep cubic (zero slopes at ends) for interpolation
    # smooth_cubic = (lambda t: 3*t*t - 2*t*t*t) if use_smooth else (lambda t: t)
    
    # grids
    eta_range = np.linspace(0.0, 1.0, eta_count)
    psi_range = np.linspace(0.0, 1.0, psi_count)
    eta_grid, psi_grid = np.meshgrid(eta_range, psi_range, indexing='ij')
    
    # intermediate arrays (raw Kulfan-style coordinates)
    x_array = np.zeros_like(eta_grid)       # axial station per column (function of psi)
    y_array = np.zeros_like(eta_grid)       # raw lateral template
    z_pos_array = np.zeros_like(eta_grid)   # raw positive vertical template
    z_neg_array = np.zeros_like(eta_grid)   # raw negative vertical template
    
    Nd = 0.005   # parameter used in your original script
    
    # Populate base templates (as in your Fig.26 reproduction)
    eps = 1e-12  # tiny clip to avoid exact zero at endpoints (safe, tiny)
    for i_eta, eta in enumerate(eta_range):
        for j_psi, psi in enumerate(psi_range):
            # =============================================================================
            # Nc = cubic(psi)
            Nc = cubic(N_in, N_out, psi)
            # =============================================================================
    
            W = 1.0
            H = 1.0
    
            Sc = 0.5**(2 * Nc)
            Cc = eta**Nc * (1 - eta)**Nc
    
            Sd = 0.5**(2 * Nd)
            psi_eff = np.clip(psi, eps, 1.0 - eps)     # avoids Cd==0 at exact endpoints
            Cd = psi_eff**Nd * (1 - psi_eff)**Nd
    
            # =============================================================================
            # x = psi * L
            x = x_centroid_inlet + psi * L
            # =============================================================================
            y = -(Sd * Cd) * (1 - 2 * eta) * W / 2.0
            z = Sd * Cd * (Sc * Cc) * H / 2.0
    
            x_array[i_eta, j_psi] = x
            y_array[i_eta, j_psi] = y
            z_pos_array[i_eta, j_psi] = z
            z_neg_array[i_eta, j_psi] = -z
    
    # Column-wise maxima used to scale each psi-column to requested absolute dimensions
    col_max_abs_y = np.max(np.abs(y_array), axis=0)            # length = psi_count
    col_max_abs_y[col_max_abs_y == 0.0] = 1.0                  # avoid div-by-zero
    
    col_max_zpos = np.max(z_pos_array, axis=0)                 # length = psi_count
    col_max_zpos[col_max_zpos == 0.0] = 1.0                    # avoid div-by-zero
    
    # width and height interpolation along x (psi index)
    # =============================================================================
    # psi_positions = psi_range * L
    psi_positions = x_centroid_inlet + psi_range * L
    # =============================================================================
    # compute w(x) and h(x) columnwise as smooth transitions
    t_vals = psi_range
    # =============================================================================
    # S_vals = smooth_cubic(t_vals)
    S_vals = cubic(0, 1, t_vals)
    # =============================================================================
    
    # =============================================================================
    w_cols = w_in + (w_out - w_in) * S_vals
    h_cols = h_in + (h_out - h_in) * S_vals
    # w_cols = w_in + (w_out - w_in) * t_vals
    # h_cols = h_in + (h_out - h_in) * t_vals
    # =============================================================================
    
    # tilt alpha(x)
    # =============================================================================
    # alpha_cols_rad = np.deg2rad( tilt_end_deg * S_vals )   # radians at each psi column
    alpha_cols_rad = np.deg2rad(cubic(tilt_start_deg, tilt_end_deg, t_vals))   # radians at each psi column
    # =============================================================================
    
    # centreline z variation z_c(x)
    # =============================================================================
    # z_centerline_cols = inlet_z + (outlet_z - inlet_z) * S_vals
    if is_inlet == True:
        alpha1_deg, alpha2_deg = angle_inlet, -angle_outlet  # upstream duct
    elif is_inlet == False:
        alpha1_deg, alpha2_deg = -angle_inlet, angle_outlet  # downstream duct
    dy0 = np.tan(np.deg2rad(alpha1_deg))   # dy/dx at left endpoint
    dy1 = np.tan(np.deg2rad(alpha2_deg))   # dy/dx at right endpoint
    z_centerline_cols = hermite_on_x(t_vals, inlet_z, outlet_z, dy0, dy1)
    # =============================================================================
    
    # prepare arrays for final rotated coordinates
    rot_X_pos = np.zeros_like(x_array)
    rot_Y_pos = np.zeros_like(y_array)
    rot_Z_pos = np.zeros_like(z_pos_array)
    
    rot_X_neg = np.zeros_like(x_array)
    rot_Y_neg = np.zeros_like(y_array)
    rot_Z_neg = np.zeros_like(z_neg_array)
    
    # Loop and place/rotate each template point columnwise
    for j in range(psi_count):
        w_j = w_cols[j]
        h_j = h_cols[j]
        alpha = alpha_cols_rad[j]
        zc = z_centerline_cols[j]
        yc = y_center
        ca = np.cos(alpha)
        sa = np.sin(alpha)
    
        # per-column scale factors so requested w_j,h_j are honoured exactly
        sy = (w_j / 2.0) / col_max_abs_y[j]    # lateral scale for column j
        sz = (h_j / 2.0) / col_max_zpos[j]     # vertical scale for column j
    
        y_col = y_array[:, j] * sy             # lateral offsets in [-w_j/2, w_j/2]
        zpos_col = z_pos_array[:, j] * sz      # positive vertical offsets ~[0, h_j/2]
        zneg_col = z_neg_array[:, j] * sz      # negative vertical offsets ~[-h_j/2, 0]
    
        # base axial for column (same for all eta)
        x_col = x_array[:, j]
    
        # rotate about local y for each eta row
        # rotated dx = sin(alpha) * z_rel ; rotated z = cos(alpha) * z_rel
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
    
    # ----------------- NEW: Extract slices that correspond to mesh columns ------------
    # Decide parametric s positions for slices along centreline.
    if include_endpoints:
        s_list = np.linspace(0.0, 1.0, num=n_slices)
    else:
        # place n_slices interior samples (exclude exact 0 and 1)
        s_list = np.linspace(0.0, 1.0, num=n_slices + 2)[1:-1]
    
    # Map s positions to column indices (nearest mesh column)
    j_indices = np.clip(np.round(s_list * (psi_count - 1)).astype(int), 0, psi_count - 1)
    
    slices_list = []  # will hold dictionaries with 's', 'psi_index', 'X','Y','Z' (closed contour)
    for s_val, j in zip(s_list, j_indices):
        # upper curve (eta increasing)
        Xu = rot_X_pos[:, j].copy()
        Yu = rot_Y_pos[:, j].copy()
        Zu = rot_Z_pos[:, j].copy()
        # lower curve (eta increasing) -> reverse it to go back from trailing to leading to close loop
        Xl = rot_X_neg[:, j].copy()
        Yl = rot_Y_neg[:, j].copy()
        Zl = rot_Z_neg[:, j].copy()
    
        # assemble closed contour: upper forward, then lower reversed
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
    # -------------------------------------------------------------------------------
    
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
    
    # OPTIONAL: print a short summary of extracted slices (first few points of each)
    print("Extracted {} slices (include_endpoints={}):".format(len(slices_list), include_endpoints))
    for i, sl in enumerate(slices_list):
        print(" slice {}: s={:.4f}, psi_index={}".format(i, sl['s'], sl['psi_index']))
        # print first 3 points
        print("  first 3 points (X,Y,Z):")
        for p in range(min(3, sl['X'].size)):
            print("   {:.4f}, {:.4f}, {:.4f}".format(sl['X'][p], sl['Y'][p], sl['Z'][p]))
            
    # =============================================================================
    last_slice = slices_list[-1]
    print(max(last_slice['Z']) - min(last_slice['Z']))
    # =============================================================================
            
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
    l_up_duct = 1 * 2  # NILS: * 2
    # HX
    l_hx = 1.5
    w_hx = 1
    h_hx = 0.75
    dx_hx_corner = 0.5
    dz_hx_corner = 0.15
    # Downstream duct
    l_down_duct = 0.5 * 4  # NILS: * 4
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
    
    # x_centroid_inlet = x_centroid_hx - l_hx/2 - l_up_duct
    # y_centroid_inlet = y_centroid_hx
    # z_centroid_inlet = z_centroid_hx + dz_hx_intake
    # x_centroid_outlet = x_centroid_hx - (l_hx/2 - dx_hx_corner) / 2
    # y_centroid_outlet = y_centroid_hx
    # z_centroid_outlet = z_centroid_hx - (h_hx/2 - dz_hx_corner) / 2
    # w_inlet = w_intake
    # h_inlet = h_intake
    # w_outlet = w_hx
    # h_outlet = (h_hx/2 + dz_hx_corner) / np.cos(np.deg2rad(90 - 22.78))
    # angle_inlet = 0.0
    # # =============================================================================
    # angle_outlet = -(90 - 22.78)
    # # alpha = 22.78
    # # beta = 115.23
    # # angle_outlet = -(180 - (alpha + beta))
    # # =============================================================================
    # N_inlet = 0.25
    # N_outlet = 0.005
    # n_slices = 3
    # include_endpoints = True
    # is_inlet = True
    
    ###
    
    x_centroid_inlet = x_centroid_hx + (l_hx/2 - dx_hx_corner) / 2
    y_centroid_inlet = y_centroid_hx
    z_centroid_inlet = z_centroid_hx + (h_hx/2 - dz_hx_corner) / 2
    x_centroid_outlet = x_centroid_hx + l_hx/2 + l_down_duct
    y_centroid_outlet = y_centroid_hx
    z_centroid_outlet = z_centroid_hx + dz_hx_fan
    w_inlet = w_hx
    h_inlet = (h_hx/2 + dz_hx_corner) / np.cos(np.deg2rad(90 - 22.78))
    w_outlet = d_fan
    h_outlet = d_fan
    angle_inlet = -(90 - 22.78)
    angle_outlet = 0.0
    N_inlet = 0.005
    N_outlet = 0.5
    n_slices = 10
    include_endpoints = False
    is_inlet = False
    
    slices_list = get_duct_geometry(
        x_centroid_inlet, y_centroid_inlet, z_centroid_inlet,
        x_centroid_outlet, y_centroid_outlet, z_centroid_outlet,
        w_inlet, h_inlet,
        w_outlet, h_outlet,
        angle_inlet, angle_outlet,
        N_inlet,
        N_outlet,
        n_slices,
        include_endpoints,
        is_inlet,
    )
    
    
    
    
    
            
            
            
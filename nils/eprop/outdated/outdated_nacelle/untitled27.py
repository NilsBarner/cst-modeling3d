#!/usr/bin/env python3
"""
CST-based nacelle generator with intuitive parameter -> Bernstein mapping.

Creates a 3D nacelle surface consistent with Kulfan (2008) style class-function +
Bernstein polynomial shape ("CST"), but accepts the intuitive parameters shown in
Kulfan Figures 3 (horizontal, 5 params) and 4 (vertical, 7 params).

Author: ChatGPT (adaptation & numerical mapper)
Date: 2026-03-07
"""

import numpy as np
from math import comb, tan, radians
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import warnings

# ------------------------
# Low-level helpers
# ------------------------
def bernstein_basis(n, psi):
    """Return Bernstein basis matrix shape (n+1, len(psi)) with B_i^n(psi)."""
    psi = np.asarray(psi)
    B = np.zeros((n + 1, psi.size))
    for i in range(n + 1):
        B[i, :] = comb(n, i) * (psi ** i) * ((1 - psi) ** (n - i))
    return B


def class_function(psi, N1=0.5, N2=1.0):
    """Kulfan class function C(psi) = psi^N1 * (1-psi)^N2."""
    psi = np.asarray(psi)
    psi = np.clip(psi, 1e-12, 1 - 1e-12)
    return (psi ** N1) * ((1 - psi) ** N2)


# ------------------------
# Parameter -> Bernstein mappers
# ------------------------
def _resolve_fraction(value, envelope):
    """
    If `value` <= 1 -> treat as fraction of envelope; else treat as meters and
    convert to nondimensional fraction by dividing by envelope.
    """
    if value is None:
        return 0.0
    value = float(value)
    if value <= 1.0:
        return value
    else:
        if envelope <= 0:
            raise ValueError("Envelope must be positive when converting absolute thickness.")
        return value / envelope


def _make_target_horizontal_zeta(psi_grid, length, width,
                                 t_max, r_le, psi_t, boat_tail_deg, closure_thickness):
    """
    Build a smooth target nondim thickness zeta(psi) for the horizontal (half-width
    nondimensionalized by width/2) from intuitive params.

    zeta defined so that physical half-width a(psi) = zeta(psi) * (width/2).
    Full local width = 2*a = zeta*width, so nondim 'full thickness' fraction = zeta.
    """
    # ensure arrays
    psi = np.asarray(psi_grid)
    # convert boat tail angle to slope (d a / dx) target at trailing edge:
    boat_rad = np.radians(boat_tail_deg)
    # we assume the half-width decreases toward TE; slope negative
    # desired da/dx (physical) approx -tan(boat_angle)
    da_dx_target = -np.tan(boat_rad)
    # convert to dzeta/dψ using a = zeta * width/2 and x = psi * length:
    # dzeta/dψ = (2/width) * da/dx * length
    dzeta_dpsi_target = (2.0 / width) * da_dx_target * length

    # curvature (2nd derivative) near psi=0 from LE radius:
    # For small x: a(x) ~ x^2 / (2 R_le)  => d2a/dx2 = 1/R_le
    # convert to dzeta/dψ^2: d2zeta/dψ2 = (2/width) * d2a/dx2 * length^2
    if r_le <= 0:
        # Large curvature -> small radius invalid; set small curvature
        d2zeta_dpsi2_target = 0.0
    else:
        d2zeta_dpsi2_target = (2.0 / width) * (1.0 / r_le) * (length ** 2)

    # Build target zeta:
    # - near psi=0 use quadratic with the desired second derivative
    # - central region: gaussian bump peaked at psi_t with amplitude t_max (nondim)
    # - tail region: linear taper to match closure value and slope approximately
    zeta = np.zeros_like(psi)

    # parameters controlling widths of regions
    le_region = min(0.12, psi_t * 0.6 + 0.02)
    tail_region = 1.0 - min(0.15, 0.35 * (1 - psi_t) + 0.02)

    # ensure closure_thickness nondim fraction already passed in
    closure_frac = closure_thickness

    # central gaussian bump
    sigma = 0.08 + 0.08 * abs(psi_t - 0.4)  # adapt width
    central_bump = t_max * np.exp(-0.5 * ((psi - psi_t) / sigma) ** 2)

    # create quadratic near LE with specified curvature:
    # zeta_le(psi) = 0.5 * d2zeta_dpsi2_target * psi^2  (since zeta'(0)=0)
    zeta_le = 0.5 * d2zeta_dpsi2_target * (psi ** 2)

    # create linear tail such that at psi=1 we get closure_frac, and near psi=1 slope is dzeta_dpsi_target
    # compute zeta at psi=tail_start using central bump value
    tail_start = max(0.75, psi_t + 0.05)
    tail_idx = psi >= tail_start
    psi_tail = psi[tail_idx]
    # slope dzeta/dψ near TE is dzeta_dpsi_target
    # linear approx: zeta(psi) = zeta(ts) + dzeta_dpsi_target*(psi - ts)
    # but enforce zeta(1)=closure_frac by adjusting the intercept slightly:
    # compute zeta at tail_start from bump:
    zeta_ts = float(np.interp(tail_start, psi, central_bump))
    # linear slope s; compute offset b0 so zeta(1)=closure_frac:
    s = dzeta_dpsi_target
    # solve for b0: closure_frac = zeta_ts + s*(1 - tail_start) + b_adjust
    b_adjust = closure_frac - (zeta_ts + s * (1 - tail_start))
    zeta_tail = zeta_ts + s * (psi_tail - tail_start) + b_adjust

    # assemble final target: take max of le/quadratic and central gaussian, then taper to tail
    zeta_candidate = np.maximum(central_bump, zeta_le)
    # where in tail region, replace with zeta_tail
    zeta_final = zeta_candidate.copy()
    zeta_final[tail_idx] = zeta_tail

    # ensure endpoint values: psi=0 => 0, psi=1 => closure_frac
    zeta_final[0] = 0.0
    zeta_final[-1] = closure_frac

    # small smoothing
    from scipy.ndimage import gaussian_filter1d
    zeta_sm = gaussian_filter1d(zeta_final, sigma=2)

    # ensure positive and bounded
    zeta_sm = np.clip(zeta_sm, 0.0, max(1.5, t_max * 1.2))

    return zeta_sm


def horizontal_params_to_bernstein(length, width, params, n_degree=4, N1=0.5, N2=1.0, zeta_T=0.0):
    """
    Map intuitive horizontal params -> Bernstein coefficients (degree n_degree).
    params = (t_max, r_le, psi_t, boat_tail_deg, closure_thickness)
    t_max, closure_thickness can be given as fraction (<=1) of width or as meters (>1).
    Returns coefficients A (length n_degree+1).
    """
    t_max_in, r_le_in, psi_t, boat_tail_deg, closure_in = params

    # convert to nondim fraction of full local width (full thickness fraction = zeta*width)
    t_max = _resolve_fraction(t_max_in, width)  # nondim
    closure_frac = _resolve_fraction(closure_in, width)

    # prevent degenerate parameters
    psi_t = float(np.clip(psi_t, 0.01, 0.99))
    r_le = float(max(r_le_in, 1e-6))

    # build psi grid for projection
    psi_grid = np.linspace(0.0, 1.0, 300)

    zeta_target = _make_target_horizontal_zeta(psi_grid, length, width,
                                               t_max, r_le, psi_t,
                                               boat_tail_deg, closure_frac)

    # Now set up linear system: unknowns are Ai in S(psi) = sum Ai * B_i(psi)
    # zeta(psi) = C(psi)*S(psi) + psi*zeta_T  => C(psi) * sum(Ai*B_i(psi)) = zeta_target - psi*zeta_T
    B = bernstein_basis(n_degree, psi_grid)  # shape (n+1, npsi)
    C = class_function(psi_grid, N1=N1, N2=N2)
    # Build matrix M of shape (npsi, ncoef) where row j = C(psi_j)*B[:,j]
    M = (C[np.newaxis, :].T * B.T)  # shape (npsi, ncoef)
    rhs = zeta_target - psi_grid * zeta_T

    # Solve least squares with Tikhonov regularization to keep coefficients moderate
    # Solve [M; sqrt(lambda)*I] A = [rhs; 0]
    lam = 1e-6  # small regularization
    MtM = M.T.dot(M) + lam * np.eye(n_degree + 1)
    MtR = M.T.dot(rhs)
    A = np.linalg.solve(MtM, MtR)

    return A


def _make_target_upper_lower_vertical(psi_grid, length, height,
                                     le_radius,
                                     up_tmax, up_psi_t, up_boat_deg,
                                     lo_tmax, lo_psi_t, lo_boat_deg):
    """
    Produce a target nondim half-height profile b_target(psi) (nondim wrt full height)
    by constructing separate upper & lower half-profiles and taking their envelope.
    Each target is constructed analogously to horizontal mapping but with its own
    peak location and boat-tail. Returns nondim half-height fraction b_target where
    physical half-height = b_target * (height/2).
    """
    psi = np.asarray(psi_grid)
    # convert params already to nondim fractions when calling this function

    # build upper target zeta_u (nondim fraction of full height)
    zeta_u = _make_target_horizontal_zeta(psi, length, height, up_tmax, le_radius, up_psi_t, up_boat_deg, 0.0)
    # build lower target zeta_l (nondim fraction of full height)
    zeta_l = _make_target_horizontal_zeta(psi, length, height, lo_tmax, le_radius, lo_psi_t, lo_boat_deg, 0.0)

    # We want a half-height envelope that accommodates both upper and lower lobes.
    # zeta_u and zeta_l are "full" nondim thickness fractions (full thickness relative to height).
    # For half-height (upper only) we take half of full local thickness if symmetric, but user provides
    # separate upper and lower maxima so we make half-height envelope as max( zeta_u/2, zeta_l/2 ).
    half_u = 0.5 * zeta_u
    half_l = 0.5 * zeta_l
    b_target = np.maximum(half_u, half_l)

    # ensure start/end conditions and smooth
    b_target[0] = 0.0
    b_target = np.clip(b_target, 0.0, 1.5)
    from scipy.ndimage import gaussian_filter1d
    b_smooth = gaussian_filter1d(b_target, sigma=2)
    return b_smooth


def vertical_params_to_bernstein(length, height, params, n_degree=6, N1=0.5, N2=1.0, zeta_T=0.0):
    """
    Map 7 intuitive vertical params to degree `n_degree` Bernstein coefficients.
    params = (r_le,
              up_tmax, up_psi_t, up_boat_deg,
              lo_tmax, lo_psi_t, lo_boat_deg)
    up_tmax, lo_tmax accept fraction of height (<=1) or absolute meters (>1).
    Returns A (length n_degree+1).
    """
    (r_le_in,
     up_tmax_in, up_psi_t, up_boat_deg,
     lo_tmax_in, lo_psi_t, lo_boat_deg) = params

    # convert thickness maxima to nondim full-height fractions
    up_t = _resolve_fraction(up_tmax_in, height)
    lo_t = _resolve_fraction(lo_tmax_in, height)

    # ensure psi_t stay in (0,1)
    up_psi_t = float(np.clip(up_psi_t, 0.01, 0.99))
    lo_psi_t = float(np.clip(lo_psi_t, 0.01, 0.99))
    r_le = float(max(r_le_in, 1e-6))

    psi_grid = np.linspace(0.0, 1.0, 300)
    # build target half-height envelope b_target (nondim relative to full height)
    b_target = _make_target_upper_lower_vertical(psi_grid, length, height,
                                                r_le,
                                                up_t, up_psi_t, up_boat_deg,
                                                lo_t, lo_psi_t, lo_boat_deg)
    # Now solve for Bernstein coefficients as before:
    B = bernstein_basis(n_degree, psi_grid)
    C = class_function(psi_grid, N1=N1, N2=N2)
    M = (C[np.newaxis, :].T * B.T)
    rhs = b_target - psi_grid * zeta_T  # include TE thickness if provided
    lam = 1e-6
    MtM = M.T.dot(M) + lam * np.eye(n_degree + 1)
    MtR = M.T.dot(rhs)
    A = np.linalg.solve(MtM, MtR)
    return A


# ------------------------
# Main generator
# ------------------------
def generate_nacelle_cst_from_intuitive(length, width, height,
                                        hor_params, vert_params,
                                        N_U=0.4, N_L=0.25,
                                        zeta_T_hor=0.0, zeta_T_vert=0.0,
                                        N1_hor=0.5, N2_hor=1.0,
                                        N1_vert=0.5, N2_vert=1.0,
                                        n_psi=100, n_eta=180,
                                        plot=True):
    """
    Main convenience function:
    - length, width, height: nacelle envelope in meters
    - hor_params: tuple/list (t_max, r_le, psi_t, boat_tail_deg, closure_thickness)
                  t_max & closure may be nondim fraction (<=1) or meters (>1)
    - vert_params: tuple/list (r_le,
                               up_tmax, up_psi_t, up_boat_deg,
                               lo_tmax, lo_psi_t, lo_boat_deg)
                   maxima may be nondim fraction (<=1) or meters (>1)
    - N_U, N_L: cross-section class exponents (scalar or array-like of length n_psi)
    Returns dictionary of mesh arrays.
    """
    # compute Bernstein coefficients from intuitive params
    A_h = horizontal_params_to_bernstein(length, width, hor_params,
                                         n_degree=4, N1=N1_hor, N2=N2_hor, zeta_T=zeta_T_hor)
    A_v = vertical_params_to_bernstein(length, height, vert_params,
                                       n_degree=6, N1=N1_vert, N2=N2_vert, zeta_T=zeta_T_vert)

    # Build axial and lateral grids
    psi = np.linspace(0.0, 1.0, n_psi)
    eta = np.linspace(0.0, 1.0, n_eta)

    # Evaluate axial CST nondim thickness zeta_h (full width fraction) and zeta_v (full height fraction)
    B_h = bernstein_basis(len(A_h) - 1, psi)
    B_v = bernstein_basis(len(A_v) - 1, psi)
    C_h = class_function(psi, N1=N1_hor, N2=N2_hor)
    C_v = class_function(psi, N1=N1_vert, N2=N2_vert)
    S_h = A_h @ B_h  # shape (len(psi),)
    S_v = A_v @ B_v
    zeta_h = C_h * S_h + psi * zeta_T_hor   # nondim full width fraction
    zeta_v = C_v * S_v + psi * zeta_T_vert  # nondim full height fraction

    # Convert nondim to physical half-dimensions
    a = zeta_h * (width / 2.0)   # half-breadth [m]
    b = zeta_v * (height / 2.0)  # half-height [m]

    # N_U / N_L handling (allow scalar or array)
    if callable(N_U):
        NU_vals = N_U(psi)
    else:
        NU_vals = np.atleast_1d(N_U)
        if NU_vals.size == 1:
            NU_vals = np.full_like(psi, float(NU_vals))
        elif NU_vals.size != psi.size:
            raise ValueError("N_U must be scalar, array of length n_psi, or callable(psi).")
    if callable(N_L):
        NL_vals = N_L(psi)
    else:
        NL_vals = np.atleast_1d(N_L)
        if NL_vals.size == 1:
            NL_vals = np.full_like(psi, float(NL_vals))
        elif NL_vals.size != psi.size:
            raise ValueError("N_L must be scalar, array of length n_psi, or callable(psi).")

    # prepare output arrays
    X = np.zeros((n_psi, n_eta))
    YU = np.zeros_like(X)
    ZU = np.zeros_like(X)
    YL = np.zeros_like(X)
    ZL = np.zeros_like(X)

    eta_clip = np.clip(eta, 1e-12, 1 - 1e-12)
    # we'll create a local unit cross-section shape S_c = 2 for "ellipse-like"
    S_c = 2.0

    for i, psi_i in enumerate(psi):
        a_i = a[i]
        b_i = b[i]
        # upper/lower class exponents
        nu = NU_vals[i]
        nl = NL_vals[i]
        Cc_u = (eta_clip ** nu) * ((1 - eta_clip) ** nu)
        Cc_l = (eta_clip ** nl) * ((1 - eta_clip) ** nl)
        # unit shape
        S_u = S_c * np.ones_like(eta)
        S_l = S_c * np.ones_like(eta)
        # nondim zeta along cross section lobes (relative to 1)
        zeta_u_eta = Cc_u * S_u
        zeta_l_eta = Cc_l * S_l
        # y coordinate (lateral): map eta in [0,1] to [-a, +a]
        y_eta = (eta - 0.5) * 2.0 * a_i
        # z (vertical): normalize by peak of Cc*Sc and scale to b_i
        # avoid division by zero
        den_u = (S_c * np.max(Cc_u) + 1e-12)
        den_l = (S_c * np.max(Cc_l) + 1e-12)
        z_u = zeta_u_eta * (b_i / den_u)
        z_l = - zeta_l_eta * (b_i / den_l)
        X[i, :] = psi_i * length
        YU[i, :] = y_eta
        YL[i, :] = y_eta
        ZU[i, :] = z_u
        ZL[i, :] = z_l

    out = {
        "x": X, "y_upper": YU, "z_upper": ZU, "y_lower": YL, "z_lower": ZL,
        "psi": psi, "eta": eta, "a_psi": a, "b_psi": b,
        "bernstein_hor": A_h, "bernstein_vert": A_v
    }

    # quick plotting
    if plot:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(out['x'], out['y_upper'], out['z_upper'],
                        rstride=2, cstride=2, linewidth=0, alpha=0.85)
        ax.plot_surface(out['x'], out['y_lower'], out['z_lower'],
                        rstride=2, cstride=2, linewidth=0, alpha=0.85)
        ax.set_xlabel('Axial x [m]')
        ax.set_ylabel('Lateral y [m]')
        ax.set_zlabel('Vertical z [m]')
        ax.set_title('Mapped CST nacelle from intuitive parameters')
        ax.view_init(elev=25, azim=130)
        plt.tight_layout()
        plt.show()

    return out


# ------------------------
# Example usage / test
# ------------------------
if __name__ == "__main__":
    # Envelope
    L = 4.0
    W = 1.2
    H = 1.2

    # Horizontal intuitive params (5):
    # - maximum thickness (here give absolute meters; <=1 would be treated as fraction)
    # - LE radius (m)
    # - location of maximum thickness psi_t (0..1)
    # - boat-tail angle at TE (deg)
    # - closure thickness at TE (m or fraction)
    horiz_params = (
        1.2,      # t_max -> we want max width equal to envelope width
        0.035,    # r_le [m]
        0.30,     # psi_t
        8.0,      # boat-tail angle (deg)
        0.01      # closure thickness [m]
    )

    # Vertical intuitive params (7):
    # - LE radius (m)
    # - upper surface: max thickness (m or frac), location psi, boat-tail angle (deg)
    # - lower surface: max thickness, location psi, boat-tail
    vert_params = (
        0.035,     # LE radius [m]
        0.6, 0.30, 8.0,   # upper: tmax (fraction=0.6 of height), psi, boat tail
        0.4, 0.40, 6.0    # lower: tmax (fraction=0.4 of height), psi, boat tail
    )

    out = generate_nacelle_cst_from_intuitive(L, W, H,
                                              horiz_params, vert_params,
                                              N_U=0.45, N_L=0.25,
                                              zeta_T_hor=0.0, zeta_T_vert=0.0,
                                              n_psi=120, n_eta=220,
                                              plot=True)

    print("Horizontal Bernstein coeffs (degree 4):", out["bernstein_hor"])
    print("Vertical Bernstein coeffs (degree 6):", out["bernstein_vert"])
    # quick checks for physical envelope:
    print("Max physical width (m):", np.max(out["a_psi"]) * 2.0)
    print("Max physical height (m):", np.max(out["b_psi"]) * 2.0)
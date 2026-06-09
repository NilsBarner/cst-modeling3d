#!/usr/bin/env python3
"""
Minimal analytic Kulfan-style nacelle generator with optional camber (no solves).

Adds small camber mean-line built from the same quadratic front/aft + class
function approach used for thickness. Camber is applied as a mean-line offset:
    Z_upper = m(psi)*length + z_upper_thickness
    Z_lower = m(psi)*length + z_lower_thickness

Author: ChatGPT (minimal modification)
"""

import numpy as np
from math import tan, radians, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _define_quadratic(x0, y0, x1, y1):
    a = (y1 - y0) / (x1 - x0) ** 2
    b = -2.0 * a * x0
    c = y0 + a * x0 ** 2
    return np.poly1d([a, b, c])


def _class_function(psi, N1=0.5, N2=1.0):
    psi = np.clip(np.asarray(psi), 1e-12, 1.0 - 1e-12)
    return psi ** N1 * (1.0 - psi) ** N2


def _resolve_fraction(value, envelope):
    """If value <= 1 treat as nondim fraction, else treat as absolute meters -> convert to fraction."""
    if value is None:
        return 0.0
    v = float(value)
    if v <= 1.0:
        return v
    else:
        return v / envelope


def generate_nacelle_from_radii_and_boattail_with_camber(
    length, width, height,
    LE_hor, TE_hor, boat_hor_deg,
    LE_ver, TE_ver, boat_ver_deg,
    N_U=0.4, N_L=0.25,
    Psi_zeta_max=0.35,
    N1=0.5, N2=1.0,
    # camber tuples: (camber_max, psi_cam) ; camber_max may be fraction<=1 or meters>1
    camber_vert=None,    # e.g. (0.02, 0.35) or (0.02*length, 0.35)
    camber_hor=None,
    n_psi=200, n_eta=160,
    plot=True
):
    """
    Minimal analytic nacelle generator with optional camber mean-line.
    """

    # nondimensional maximum radii (per the earlier convention: radius = zeta * length)
    zeta_max_hor = float(width) / (2.0 * float(length))
    zeta_max_ver = float(height) / (2.0 * float(length))

    # helper: build zeta(psi) as before
    def _build_zeta_profile(R_le, R_te, beta_deg, zeta_max):
        R_le_over_c = max(R_le / length, 1e-12)
        S_le = sqrt(2.0 * R_le_over_c)
        zeta_te = max(R_te / length, 0.0)
        S_te = tan(radians(beta_deg)) + zeta_te
        S_zeta_max = (zeta_max - Psi_zeta_max * zeta_te) / (np.sqrt(Psi_zeta_max) * (1.0 - Psi_zeta_max))
        quadratic_front = _define_quadratic(Psi_zeta_max, S_zeta_max, 0.0, S_le)
        quadratic_aft = _define_quadratic(Psi_zeta_max, S_zeta_max, 1.0, S_te)
        psi = np.linspace(1e-6, 1.0, n_psi)
        S_list = np.where(psi >= Psi_zeta_max, quadratic_aft(psi), quadratic_front(psi))
        C = _class_function(psi, N1=N1, N2=N2)
        zeta = C * S_list + psi * zeta_te
        zeta[0] = 0.0
        zeta[-1] = zeta_te
        return psi, zeta

    # helper: build camber mean-line (nondim, relative to length)
    def build_camber_profile(camber_tuple, envelope):
        """
        Minimal camber builder that uses the same quadratic-front/aft + class function approach.
        camber_tuple = (camber_max, psi_cam)
        camber_max may be fraction<=1 (of envelope) or meters>1 and will be converted.
        Returns psi, camber_nondim (so physical mean-line m(psi)*length in meters).
        """
        if camber_tuple is None:
            psi = np.linspace(0.0, 1.0, n_psi)
            return psi, np.zeros_like(psi)

        camber_max_in, psi_cam = camber_tuple
        # convert to nondim fraction relative to length (consistent units with zeta)
        # user may have passed camber in meters or as fraction of envelope (<=1).
        # here envelope = length (we treat meanline nondim as fraction of length)
        camber_max = _resolve_fraction(camber_max_in, length)

        # define pseudo LE and TE 'camber' end-values: set camber_te=0 for simplicity
        camber_te = 0.0
        S_le = 0.0  # camber at psi=0 we want zero slope typically; use 0 for simplicity
        S_te = 0.0  # enforce zero camber at TE for smooth closure; could be nonzero if desired

        # construct S_camber_max analogously (avoid division by zero)
        Psi_c = float(np.clip(psi_cam, 1e-4, 1.0 - 1e-4))
        S_camber_max = (camber_max - Psi_c * camber_te) / (np.sqrt(Psi_c) * (1.0 - Psi_c) + 1e-12)

        quad_front = _define_quadratic(Psi_c, S_camber_max, 0.0, S_le)
        quad_aft = _define_quadratic(Psi_c, S_camber_max, 1.0, S_te)

        psi = np.linspace(1e-6, 1.0, n_psi)
        S_list = np.where(psi >= Psi_c, quad_aft(psi), quad_front(psi))
        C = _class_function(psi, N1=N1, N2=N2)
        camber = C * S_list + psi * camber_te
        camber[0] = 0.0
        camber[-1] = camber_te
        return psi, camber

    # Build base zeta profiles
    psi_h, zeta_h = _build_zeta_profile(LE_hor, TE_hor, boat_hor_deg, zeta_max_hor)
    psi_v, zeta_v = _build_zeta_profile(LE_ver, TE_ver, boat_ver_deg, zeta_max_ver)

    # Build camber profiles (nondim relative to length)
    psi_cam_h, camber_h = build_camber_profile(camber_hor, length)
    psi_cam_v, camber_v = build_camber_profile(camber_vert, length)

    # unify psi grids if necessary
    if psi_h.size != psi_v.size or not np.allclose(psi_h, psi_v):
        psi = np.linspace(0.0, 1.0, n_psi)
        zeta_h = np.interp(psi, psi_h, zeta_h)
        zeta_v = np.interp(psi, psi_v, zeta_v)
        camber_h = np.interp(psi, psi_cam_h, camber_h)
        camber_v = np.interp(psi, psi_cam_v, camber_v)
    else:
        psi = psi_h
        # align cambers if same grid
        camber_h = np.interp(psi, psi_cam_h, camber_h)
        camber_v = np.interp(psi, psi_cam_v, camber_v)

    # physical half-dimensions (meters) using your earlier convention:
    a = zeta_h * length      # half-width [m]
    b = zeta_v * length      # half-height [m]

    # mean-line physical offset (meters)
    m_h = camber_h * length
    m_v = camber_v * length

    # N_U / N_L handling
    NU_vals = np.atleast_1d(N_U)
    if NU_vals.size == 1:
        NU_vals = np.full_like(psi, float(NU_vals))
    elif NU_vals.size != psi.size:
        raise ValueError("N_U must be scalar or array of length n_psi")
    NL_vals = np.atleast_1d(N_L)
    if NL_vals.size == 1:
        NL_vals = np.full_like(psi, float(NL_vals))
    elif NL_vals.size != psi.size:
        raise ValueError("N_L must be scalar or array of length n_psi")

    # Build mesh (vertical camber used here as mean-line offset in z; horizontal camber would shift y center)
    eta = np.linspace(0.0, 1.0, n_eta)
    eta_clip = np.clip(eta, 1e-12, 1.0 - 1e-12)
    S_c = 2.0

    X = np.zeros((n_psi, n_eta))
    YU = np.zeros_like(X)
    ZU = np.zeros_like(X)
    YL = np.zeros_like(X)
    ZL = np.zeros_like(X)

    for i, psi_i in enumerate(psi):
        a_i = a[i]
        b_i = b[i]
        mvi = m_v[i]   # vertical mean-line offset [m]
        mui = m_h[i]   # horizontal mean-line offset (if used, here unused for y-centering)

        nu = NU_vals[i]
        nl = NL_vals[i]
        Cc_u = (eta_clip ** nu) * ((1.0 - eta_clip) ** nu)
        Cc_l = (eta_clip ** nl) * ((1.0 - eta_clip) ** nl)
        S_u = S_c * np.ones_like(eta)
        S_l = S_c * np.ones_like(eta)
        zeta_u_eta = Cc_u * S_u
        zeta_l_eta = Cc_l * S_l

        # lateral y from -a to +a (we keep horizontal camber only as a mean offset if desired)
        y_eta = (eta - 0.5) * 2.0 * a_i

        den_u = (S_c * np.max(Cc_u) + 1e-12)
        den_l = (S_c * np.max(Cc_l) + 1e-12)
        # thickness components (symmetric about centerline before camber)
        z_u = zeta_u_eta * (b_i / den_u)
        z_l = - zeta_l_eta * (b_i / den_l)

        # apply vertical camber mean-line: shift both upper and lower z by m_v[psi]
        ZU[i, :] = mvi + z_u
        ZL[i, :] = mvi + z_l

        # (optional) if you wanted horizontal camber to shift y centerline:
        # YU[i, :] = mui + y_eta
        # YL[i, :] = mui + y_eta
        # but we keep horizontal camber off by default and use same lateral coordinates:
        YU[i, :] = y_eta
        YL[i, :] = y_eta

        X[i, :] = psi_i * length

    out = {
        "x": X, "y_upper": YU, "z_upper": ZU, "y_lower": YL, "z_lower": ZL,
        "psi": psi, "eta": eta,
        "zeta_h": zeta_h, "zeta_v": zeta_v,
        "a_psi_m": a, "b_psi_m": b,
        "camber_h_nondim": camber_h, "camber_v_nondim": camber_v,
        "camber_h_m": m_h, "camber_v_m": m_v
    }

    if plot:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(out['x'], out['y_upper'], out['z_upper'],
                        rstride=max(1, n_psi // 80), cstride=max(1, n_eta // 60),
                        linewidth=0, alpha=0.9)
        ax.plot_surface(out['x'], out['y_lower'], out['z_lower'],
                        rstride=max(1, n_psi // 80), cstride=max(1, n_eta // 60),
                        linewidth=0, alpha=0.9)
        ax.set_xlabel('Axial x [m]')
        ax.set_ylabel('Lateral y [m]')
        ax.set_zlabel('Vertical z [m]')
        ax.set_title('Nacelle with optional camber (analytic)')
        ax.set_aspect('equal')
        ax.view_init(elev=22, azim=130)
        plt.tight_layout()
        plt.show()

    return out


# ---------------------------
# Quick demo
# ---------------------------
if __name__ == "__main__":
    L = 4.0
    W = 1.2 * 0.8
    H = 1.2 * 1.2

    LE_h = 0.1 * 2  # 0.035
    TE_h = 0.15 * 2  # 0.015
    boat_h = 8.0

    LE_v = 0.1 * 3  # 0.035
    TE_v = 0.1 * 3  # 0.01
    boat_v = 6.0

    # Example: small vertical camber of 0.01 m (peak) located at psi=0.35
    res = generate_nacelle_from_radii_and_boattail_with_camber(
        L, W, H,
        LE_h, TE_h, boat_h,
        LE_v, TE_v, boat_v,
        N_U=0.45, N_L=0.25,
        # camber_vert=(-0.01 * 4, 0.35),  # 1 cm peak camber
        camber_vert=None,
        camber_hor=None,
        n_psi=200, n_eta=200, plot=True
    )

    print("Max physical width (m):", 2.0 * np.max(res["a_psi_m"]))
    print("Max physical height (m):", 2.0 * np.max(res["b_psi_m"]))
    print("Max vertical camber (m):", np.max(res["camber_v_m"]))
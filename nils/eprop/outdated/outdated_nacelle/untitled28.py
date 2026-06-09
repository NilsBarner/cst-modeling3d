#!/usr/bin/env python3
"""
Analytic Kulfan-style nacelle (3D) generator — no systems, direct formulae only.

Function:
    generate_nacelle_from_radii_and_boattail_simple(
        length, width, height,
        LE_hor, TE_hor, boat_hor_deg,
        LE_ver, TE_ver, boat_ver_deg,
        N_U=0.4, N_L=0.25,
        Psi_zeta_max=0.35,
        N1=0.5, N2=1.0,
        n_psi=200, n_eta=160,
        plot=True
    )

Notes:
- Uses the same nondimensional ζ definition as in your provided script:
    physical half-width (m) = ζ(ψ) * length.
  Hence ζ_max = width / (2*length) and ζ_max_vert = height / (2*length).
- ζ_te is approximated as R_TE / length (simple local scale).
- All transforms are closed-form evaluations and simple polynomial tangents.
"""
import numpy as np
from math import tan, radians, sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def _define_quadratic(x0, y0, x1, y1):
    """
    Quadratic ax^2 + bx + c that is tangent at (x0,y0) (horizontal tangent)
    and passes through (x1,y1) (secant condition), matching your earlier helper.
    """
    a = (y1 - y0) / (x1 - x0) ** 2
    b = -2.0 * a * x0
    c = y0 + a * x0 ** 2
    return np.poly1d([a, b, c])


def _class_function(psi, N1=0.5, N2=1.0):
    psi = np.clip(np.asarray(psi), 1e-12, 1 - 1e-12)
    return psi ** N1 * (1.0 - psi) ** N2


def generate_nacelle_from_radii_and_boattail_simple(
    length, width, height,
    LE_hor, TE_hor, boat_hor_deg,
    LE_ver, TE_ver, boat_ver_deg,
    N_U=0.4, N_L=0.25,
    Psi_zeta_max=0.35,
    N1=0.5, N2=1.0,
    n_psi=200, n_eta=160,
    plot=True
):
    """
    Analytic (no linear systems) nacelle generator following your example code.
    Inputs:
      - length, width, height : overall envelope (m)
      - LE_hor, TE_hor : leading/trailing edge radii (m) in horizontal plane
      - boat_hor_deg : boat-tail angle at trailing edge (deg) in horizontal plane
      - LE_ver, TE_ver, boat_ver_deg : same in vertical plane
      - N_U, N_L : upper/lower cross-section class exponents (scalar or arrays length n_psi)
      - Psi_zeta_max : location of maximum nondim radius (default 0.35)
      - N1, N2 : class function exponents (defaults for round nose)
      - n_psi, n_eta : resolution
    Returns:
      dict with mesh arrays (x, y_upper, z_upper, y_lower, z_lower), axial psi, eta,
      and the nondimensional zeta profiles for both planes.
    """

    # ---------------------------------------------------------------------
    # Translate physical envelopes to nondimensional max-radius (ζ_max)
    # ζ_max is nondimensional radius relative to length: radius = ζ * length
    # So width = 2 * ζ_max * length -> ζ_max = width/(2*length)
    # ---------------------------------------------------------------------
    zeta_max_hor = float(width) / (2.0 * float(length))
    zeta_max_ver = float(height) / (2.0 * float(length))

    # A helper building zeta(psi) from LE radius, TE radius and boat tail angle
    def _build_zeta_profile(R_le, R_te, beta_deg, zeta_max, plane_name="hor"):
        """
        R_le, R_te in meters, zeta_max nondim (radius/length).
        Returns psi array and zeta(psi) nondim radius (half-radius relative to length).
        """

        # nondimensional LE radius for Kulfan formula ~ R_le / c
        R_le_over_c = max(R_le / length, 1e-12)
        S_le = sqrt(2.0 * R_le_over_c)  # Kulfan-like formula (Eq. 4 style used in your script)

        # estimate zeta_te (nondimensional radius at TE) from R_te
        # simple mapping: zeta_te = R_te / length  (assumption: small TE radius relative to length)
        zeta_te = max(R_te / length, 0.0)

        # S_te set as tan(beta_tail) + zeta_te (your script's eq. (5) style)
        S_te = tan(radians(beta_deg)) + zeta_te

        # S_zeta_max from equation (2) in your example:
        # S_zeta_max = (zeta_max - Psi_zeta_max * zeta_te) / (sqrt(Psi_zeta_max) * (1 - Psi_zeta_max))
        S_zeta_max = (zeta_max - Psi_zeta_max * zeta_te) / (np.sqrt(Psi_zeta_max) * (1.0 - Psi_zeta_max))

        # Build quadratic polynomials front & aft tangent at (Psi_zeta_max, S_zeta_max)
        quadratic_front = _define_quadratic(Psi_zeta_max, S_zeta_max, 0.0, S_le)
        quadratic_aft = _define_quadratic(Psi_zeta_max, S_zeta_max, 1.0, S_te)

        # create psi grid and evaluate S(psi)
        psi = np.linspace(1e-6, 1.0, n_psi)
        S_list = np.zeros_like(psi)
        for i, p in enumerate(psi):
            if p >= Psi_zeta_max:
                S_list[i] = quadratic_aft(p)
            else:
                S_list[i] = quadratic_front(p)

        # class function
        C = _class_function(psi, N1=N1, N2=N2)

        # zeta nondim
        zeta = C * S_list + psi * zeta_te

        # Ensure endpoints consistent: psi=0 -> 0, psi=1 -> zeta_te (or close)
        zeta[0] = 0.0
        zeta[-1] = zeta_te

        return psi, zeta

    # Build horizontal and vertical ζ profiles
    psi_h, zeta_h = _build_zeta_profile(LE_hor, TE_hor, boat_hor_deg, zeta_max_hor, "horizontal")
    psi_v, zeta_v = _build_zeta_profile(LE_ver, TE_ver, boat_ver_deg, zeta_max_ver, "vertical")

    # Validate psi arrays equal length; if different, re-sample to common psi
    if psi_h.size != psi_v.size:
        # resample to n_psi evenly
        psi = np.linspace(0.0, 1.0, n_psi)
        zeta_h = np.interp(psi, psi_h, zeta_h)
        zeta_v = np.interp(psi, psi_v, zeta_v)
    else:
        psi = psi_h

    # convert nondim zeta to physical half-dimensions (meters) using same convention as your script:
    # physical half-width a(psi) [m] = zeta_h(psi) * length
    a = zeta_h * length
    b = zeta_v * length

    # Support N_U / N_L as scalars or arrays (like earlier)
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

    # Build 3D mesh using the cross-section lobes approach from your first script
    eta = np.linspace(0.0, 1.0, n_eta)
    eta_clip = np.clip(eta, 1e-12, 1.0 - 1e-12)
    S_c = 2.0  # baseline shape magnitude (ellipse-like normalization used earlier)

    X = np.zeros((n_psi, n_eta))
    YU = np.zeros_like(X)
    ZU = np.zeros_like(X)
    YL = np.zeros_like(X)
    ZL = np.zeros_like(X)

    for i, psi_i in enumerate(psi):
        a_i = a[i]    # physical half-width (m)
        b_i = b[i]    # physical half-height (m)
        nu = NU_vals[i]
        nl = NL_vals[i]

        Cc_u = (eta_clip ** nu) * ((1.0 - eta_clip) ** nu)
        Cc_l = (eta_clip ** nl) * ((1.0 - eta_clip) ** nl)
        S_u = S_c * np.ones_like(eta)
        S_l = S_c * np.ones_like(eta)
        zeta_u_eta = Cc_u * S_u
        zeta_l_eta = Cc_l * S_l

        # lateral coordinate from -a to +a
        y_eta = (eta - 0.5) * 2.0 * a_i

        # vertical coordinates, normalized by max( Cc * S_c ) then scaled to b_i
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
        "psi": psi, "eta": eta,
        "zeta_h": zeta_h, "zeta_v": zeta_v,
        "a_psi_m": a, "b_psi_m": b,
        "zeta_max_hor": zeta_max_hor, "zeta_max_ver": zeta_max_ver
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
        ax.set_title('Nacelle (simple analytic Kulfan-style mapping)')
        ax.set_aspect('equal')
        ax.view_init(elev=22, azim=130)
        plt.tight_layout()
        plt.show()

    return out


# ---------------------------
# Demo / quick check
# ---------------------------
if __name__ == "__main__":
    L = 4.0
    W = 1.2 * 1.5
    H = 1.2 * 1.5

    LE_h = 0.1 * 2  # 0.035
    TE_h = 0.15 * 4  # 0.015
    boat_h = 8.0

    LE_v = 0.1 * 4  # 0.035
    TE_v = 0.1 * 4  # 0.01
    boat_v = 6.0

    res = generate_nacelle_from_radii_and_boattail_simple(
        L, W, H,
        LE_h, TE_h, boat_h,
        LE_v, TE_v, boat_v,
        N_U=0.45, N_L=0.25,
        Psi_zeta_max=0.35,
        n_psi=200, n_eta=200, plot=True
    )

    print("max physical width (m):", 2.0 * np.max(res["a_psi_m"]))  # should be ≈ W
    print("max physical height (m):", 2.0 * np.max(res["b_psi_m"]))  # should be ≈ H
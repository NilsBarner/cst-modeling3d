import numpy as np
from math import comb
import matplotlib.pyplot as plt

def _bernstein_shape(A, psi):
    """
    Evaluate shape function S(psi) given Bernstein coefficients A = [A0..An].
    Uses S(psi) = sum_{i=0..n} A[i] * C(n,i) * psi^i * (1-psi)^(n-i)
    """
    A = np.asarray(A)
    n = A.size - 1
    psi = np.asarray(psi)
    S = np.zeros_like(psi, dtype=float)
    for i, Ai in enumerate(A):
        K = comb(n, i)
        S += Ai * K * (psi**i) * ((1 - psi)**(n - i))
    return S

def _class_function(psi, N1=0.5, N2=1.0):
    """C(psi) = psi**N1 * (1-psi)**N2"""
    psi = np.asarray(psi)
    # protect endpoints slightly
    psi = np.clip(psi, 1e-12, 1 - 1e-12)
    return (psi**N1) * ((1 - psi)**N2)

def generate_nacelle_cst(length, width, height,
                         hor_coeffs, vert_coeffs,
                         N_U=0.5, N_L=0.25,
                         zeta_T_hor=0.0, zeta_T_vert=0.0,
                         N1_hor=0.5, N2_hor=1.0,
                         N1_vert=0.5, N2_vert=1.0,
                         n_psi=80, n_eta=120,
                         plot=True):
    """
    Generate a 3D nacelle surface using CST/class-function fundamentals (Kulfan 2008).
    Parameters
    ----------
    length, width, height : floats (m) overall dimensions
    hor_coeffs : array-like (m+1 floats) -- Bernstein coefficients for horizontal thickness S_h(psi)
                 (e.g. length 5 for a 5-parameter symmetric example -> order 4)
    vert_coeffs : array-like (k+1 floats) -- Bernstein coeffs for vertical cambered S_v(psi)
    N_U, N_L : scalar or array-like of length n_psi or callable(psi)
               class-function exponents for upper / lower cross-section lobes (can vary with psi)
    zeta_T_hor, zeta_T_vert : trailing-edge thickness terms (nondim ratios)
    N1_hor, N2_hor, N1_vert, N2_vert : class function exponents for axial CST (defaults for round-nose)
    n_psi, n_eta : int, sampling resolution
    plot : bool -- show a matplotlib 3D surface plot
    Returns
    -------
    dict with mesh arrays: { 'x':X, 'y_upper':YU, 'z_upper':ZU, 'y_lower':YL, 'z_lower':ZL,
                            'psi':psi, 'eta':eta, 'a_psi':a, 'b_psi':b }
    Notes
    -----
    - The axial coordinate psi runs 0..1 and is scaled by `length`. Horizontal half-width a(psi)=hor_thickness(psi)*width/2.
      Vertical half-height b(psi)=vert_thickness(psi)*height/2.
    - This routine uses a unit-shape cross-section baseline (S_c=2 for ellipse-like lobes) and applies class
      functions with exponents N_U and N_L to shape the lobes (consistent with Kulfan §VIII and §X). See paper.
    """
    # axial param
    psi = np.linspace(0.0, 1.0, n_psi)
    eta = np.linspace(0.0, 1.0, n_eta)

    # evaluate axial shape functions (Bernstein)
    S_h = _bernstein_shape(hor_coeffs, psi)        # horizontal shape function S(psi)
    S_v = _bernstein_shape(vert_coeffs, psi)       # vertical shape function S(psi)

    # class functions along axial direction
    C_h = _class_function(psi, N1=N1_hor, N2=N2_hor)
    C_v = _class_function(psi, N1=N1_vert, N2=N2_vert)

    # nondimensional thickness distributions (half-thickness ratios)
    zeta_h = C_h * S_h + psi * zeta_T_hor   # horizontal nondim half-width * 2 maybe depending on S definition
    zeta_v = C_v * S_v + psi * zeta_T_vert

    # convert to physical half-dimensions
    a = zeta_h * (width / 2.0)    # half-breadth (y direction)
    b = zeta_v * (height / 2.0)   # half-height (z direction)

    # allow N_U / N_L to be callable or arrays
    if callable(N_U):
        NU_vals = N_U(psi)
    else:
        NU_vals = np.asarray(N_U)
        if NU_vals.size == 1:
            NU_vals = np.full_like(psi, float(NU_vals))
        elif NU_vals.size != psi.size:
            raise ValueError("N_U must be scalar, array of length n_psi, or callable(psi).")

    if callable(N_L):
        NL_vals = N_L(psi)
    else:
        NL_vals = np.asarray(N_L)
        if NL_vals.size == 1:
            NL_vals = np.full_like(psi, float(NL_vals))
        elif NL_vals.size != psi.size:
            raise ValueError("N_L must be scalar, array of length n_psi, or callable(psi).")

    # Prepare output grids
    X = np.zeros((n_psi, n_eta))
    YU = np.zeros_like(X)
    ZU = np.zeros_like(X)
    YL = np.zeros_like(X)
    ZL = np.zeros_like(X)

    # unit cross-section shape function baseline (use 2 for ellipse-like lobes as in Kulfan)
    S_c_unit = 2.0

    # Build mesh
    for i, psi_i in enumerate(psi):
        a_i = a[i]
        b_i = b[i]
        # cross-section class functions for this psi
        NC1 = NU_vals[i]
        NC2 = NU_vals[i]
        # upper lobe class function Cc_upper(eta)
        eta_clip = np.clip(eta, 1e-12, 1 - 1e-12)
        Cc_upper = (eta_clip**NC1) * ((1 - eta_clip)**NC2)
        # lower lobe class function (use NL for both exponents for simplicity)
        NC1l = NL_vals[i]
        NC2l = NL_vals[i]
        Cc_lower = (eta_clip**NC1l) * ((1 - eta_clip)**NC2l)

        # shape functions for cross-section: here we use unit baseline S_c_unit
        S_upper = S_c_unit * np.ones_like(eta)
        S_lower = S_c_unit * np.ones_like(eta)

        # nondim zeta along cross-section lobes
        zeta_u_eta = Cc_upper * S_upper
        zeta_l_eta = Cc_lower * S_lower

        # Map to y and z
        # y: lateral coordinate from -a to +a
        y_eta = (eta - 0.5) * 2.0 * a_i

        # z: upper z = + zeta_u_eta * b_i / (max(S_c_unit*C?) ) - but S_c_unit already baseline, just scale
        z_u = zeta_u_eta * (b_i / (S_c_unit * np.max(Cc_upper) + 1e-12))
        z_l = - zeta_l_eta * (b_i / (S_c_unit * np.max(Cc_lower) + 1e-12))

        # fill mesh row
        X[i, :] = psi_i * length
        YU[i, :] = y_eta
        YL[i, :] = y_eta
        ZU[i, :] = z_u
        ZL[i, :] = z_l

    out = dict(x=X, y_upper=YU, z_upper=ZU, y_lower=YL, z_lower=ZL,
               psi=psi, eta=eta, a_psi=a, b_psi=b)

    if plot:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        fig = plt.figure(figsize=(10,7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(out['x'], out['y_upper'], out['z_upper'],
                        rstride=2, cstride=2, linewidth=0, alpha=0.85)
        ax.plot_surface(out['x'], out['y_lower'], out['z_lower'],
                        rstride=2, cstride=2, linewidth=0, alpha=0.85)
        ax.set_xlabel('Axial x [m]')
        ax.set_ylabel('Lateral y [m]')
        ax.set_zlabel('Vertical z [m]')
        ax.set_title('CST nacelle (axial param psi -> x)')
        ax.set_aspect('equal')
        ax.view_init(elev=20, azim=120)
        plt.tight_layout()
        plt.show()

    return out

# -------------------------
# Example call (quick demo)
# -------------------------
if __name__ == "__main__":
    L, W, H = 4.0, 1.2, 1.2  # m
    # Example Bernstein coefficients:
    # - horizontal: 5 numbers (order 4 / BPO4)
    hor_coeffs = [0.2, 0.8, 0.6, 0.3, 0.01]  # tuneable: endpoint values control LE radius and TE closure
    # - vertical: 7 numbers (order 6 / BPO6)
    vert_coeffs = [0.3, 0.9, 0.7, 0.45, 0.3, 0.12, 0.01]

    result = generate_nacelle_cst(L, W, H,
                                 hor_coeffs, vert_coeffs,
                                 N_U=0.5, N_L=0.25,
                                 zeta_T_hor=0.0, zeta_T_vert=0.0,
                                 n_psi=80, n_eta=160,
                                 plot=True)
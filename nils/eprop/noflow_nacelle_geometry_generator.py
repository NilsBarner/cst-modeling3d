"""
This script parametrises and plots a turboprop-like (I call it no-flow) nacelle
with spinner using Kulfan's shape space methods described in
low_paper_2008_univparamgeomreprmeth_kulfan.

The nondimensional coordinates are psi (x), eta (y), and zeta (z).

Minimal analytic Kulfan-style nacelle generator with optional camber (no solves).

Adds small camber mean-line built from the same quadratic front/aft + class
function approach used for thickness. Camber is applied as a mean-line offset:
    Z_upper = m(psi)*length + z_upper_thickness
    Z_lower = m(psi)*length + z_lower_thickness

Minimal spinner added: see inline comments.
"""

__all__ = ["generate_noflow_nacelle_geometry"]

import sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from math import tan, radians, sqrt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.append(r'C:\Users\nmb48\Documents\GitHub\cst-modeling3d')
from cst_modeling.basic import BasicSurface

def _quadratic(x0, y0, x1, y1):
    a = (y1 - y0) / (x1 - x0) ** 2
    b = -2.0 * a * x0
    c = y0 + a * x0 ** 2
    return np.poly1d([a, b, c])


def _class_function(psi, N1=0.5, N2=1.0):
    psi = np.clip(np.asarray(psi), 1e-12, 1.0 - 1e-12)
    return psi ** N1 * (1.0 - psi) ** N2


# helper: build zeta(psi) as before
def _build_zeta_profile(
    R_le, R_te, beta, zeta_max,
    length,
    Psi_zeta_max,
    N1, N2,
    N_psi,
):
    R_le_over_c = max(R_le / length, 1e-12)
    S_le = sqrt(2.0 * R_le_over_c)
    zeta_te = max(R_te / length, 0.0)
    S_te = tan(beta) + zeta_te
    S_zeta_max = (zeta_max - Psi_zeta_max * zeta_te) / (np.sqrt(Psi_zeta_max) * (1.0 - Psi_zeta_max))
    quadratic_front = _quadratic(Psi_zeta_max, S_zeta_max, 0.0, S_le)
    quadratic_aft = _quadratic(Psi_zeta_max, S_zeta_max, 1.0, S_te)
    psi = np.linspace(1e-6, 1.0, N_psi)
    S_list = np.where(psi >= Psi_zeta_max, quadratic_aft(psi), quadratic_front(psi))
    C = _class_function(psi, N1=N1, N2=N2)
    zeta = C * S_list + psi * zeta_te
    zeta[0] = 0.0
    zeta[-1] = zeta_te
    return psi, zeta

# helper: build camber mean-line (nondim, relative to length)
def build_camber_profile(psi, camber_tuple, envelope, N1, N2, N_psi):
    """
    Minimal camber builder that uses the same quadratic-front/aft + class function approach.
    camber_tuple = (camber_max, psi_cam)
    camber_max may be fraction<=1 (of envelope) or meters>1 and will be converted.
    Returns psi, camber_nondim (so physical mean-line m(psi)*length in meters).
    """

    camber_max_in, psi_cam = camber_tuple
    # convert to nondim fraction relative to length (consistent units with zeta)
    # user may have passed camber in meters or as fraction of envelope (<=1).
    # here envelope = length (we treat meanline nondim as fraction of length)
    camber_max = camber_max_in  # _resolve_fraction(camber_max_in, length)

    # define pseudo LE and TE 'camber' end-values: set camber_te=0 for simplicity
    camber_te = 0.0
    S_le = 0.0  # camber at psi=0 we want zero slope typically; use 0 for simplicity
    S_te = 0.0  # enforce zero camber at TE for smooth closure; could be nonzero if desired

    # construct S_camber_max analogously (avoid division by zero)
    Psi_c = float(np.clip(psi_cam, 1e-4, 1.0 - 1e-4))
    S_camber_max = (camber_max - Psi_c * camber_te) / (np.sqrt(Psi_c) * (1.0 - Psi_c) + 1e-12)

    quad_front = _quadratic(Psi_c, S_camber_max, 0.0, S_le)
    quad_aft = _quadratic(Psi_c, S_camber_max, 1.0, S_te)

    # psi = np.linspace(1e-6, 1.0, N_psi)
    S_list = np.where(psi >= Psi_c, quad_aft(psi), quad_front(psi))
    C = _class_function(psi, N1=N1, N2=N2)
    camber = C * S_list + psi * camber_te
    camber[0] = 0.0
    camber[-1] = camber_te
    return psi, camber


def generate_spinner_geometry(
    w_le,
    h_half, psi, spinner_length_frac, length,
    N_eta,
    dx=0.0,
    dy=0.0,
    dz=0.0,
):
    
    # Spinner END located where vertical nacelle half-height b(psi) = w_le.
    # Spinner then protrudes forward by spinner_length_frac * length.
    
    # Find axial station where vertical half-height equals LE radius
    b_array = np.asarray(h_half)
    psi_array = np.asarray(psi)
    
    psi_attach = float(np.interp(w_le, b_array, psi_array))

    # Spinner protrusion length (user input fraction of nacelle length)
    c_sp = float(spinner_length_frac) * length

    # Spinner aft end (attachment point)
    x_attach = psi_attach * length

    # Spinner tip location (may be < 0)
    x_tip = x_attach - c_sp
    
    # shape-space cubic S_hat (Eq. C.5)
    def S_hat(psih):
        return -0.8408 * (psih ** 3) + 1.2136 * (psih ** 2) - 0.1941 * psih + 0.4

    n_spi = max(40, int(80 * spinner_length_frac))
    psih = np.linspace(0.0, 1.0, n_spi)

    # Use Hawkswell shape-space mapping (Eq. C.1 & C.5)
    Svals = S_hat(psih)
    zetate = S_hat(1.0)

    zeta_hat = Svals * np.sqrt(psih) * (1.0 - psih) + psih * zetate

    r_raw = zeta_hat * c_sp

    r_scale = w_le / float(r_raw.max())

    r_vals = r_raw * r_scale

    # build axisymmetric mesh
    n_theta = max(40, N_eta)
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta)

    R_mesh = np.outer(r_vals, np.cos(theta)) + dy
    Z_mesh = np.outer(r_vals, np.sin(theta)) + dz

    # IMPORTANT: spinner now runs from x_tip → x_attach
    X_mesh = np.outer(x_tip + psih * c_sp, np.ones(n_theta)) + dx
    
    vtk_dataset = pv.StructuredGrid(X_mesh, R_mesh, Z_mesh)  # create VTK dataset from surface coordinates

    spinner_dict = {
        "psi_attach": psi_attach,
        "x_attach": x_attach,
        "x_tip": x_tip,
        "c_sp": c_sp,
        "psih": psih,
        "theta": theta,
        "r_vals": r_vals,
        "x_mesh": X_mesh,
        "y_mesh": R_mesh,
        "z_mesh": Z_mesh,
        "vtk_dataset": vtk_dataset,
    }
    
    return spinner_dict


def generate_noflow_nacelle_geometry(
    length, width, height,
    w_le, w_te, beta_horz,
    h_le, h_te, beta_vert,
    N_u=0.4, N_l=0.25,
    Psi_zeta_max=0.35,
    N1=0.5, N2=1.0,
    camber_vert_tuple=None,
    spinner_length_frac=0.1,
    N_psi=200, N_eta=160,
    plot=True,
    write_dat_file=False,
    dx=0.0,
    dy=0.0,
    dz=0.0,
):
    """
    Minimal analytic nacelle generator with optional camber mean-line.
    """

    # Nondimensional maximum profile thicknesses
    zeta_max_horz = float(width) / (2.0 * float(length))
    zeta_max_vert = float(height) / (2.0 * float(length))
    
    # Nondimensional thickness profiles
    psi, zeta_horz = _build_zeta_profile(
        w_le, w_te, beta_horz, zeta_max_horz,
        length,
        Psi_zeta_max,
        N1, N2,
        N_psi,
    )
    _, zeta_vert = _build_zeta_profile(
        h_le, h_te, beta_vert, zeta_max_vert,
        length,
        Psi_zeta_max,
        N1, N2,
        N_psi,
    )

    # Physical half-thicknesses
    w_half = zeta_horz * length  # half-width
    h_half = zeta_vert * length  # half-height

    # Nondimensional vertical camber profile
    psi_camber_vert, zeta_camber_vert = build_camber_profile(
        psi,
        camber_vert_tuple, length, N1, N2,
        N_psi,
    )
    
    # Physical mean-line offset
    dz_meanline = zeta_camber_vert * length
    
    # Generate spinner geometry
    spinner_dict = generate_spinner_geometry(
        # w_le / 1.5,
        0.45,
        h_half, psi, spinner_length_frac, length,
        N_eta,
        dx,
        dy,
        dz,
    )
    
    # N_u / N_l handling
    N_u_array = np.atleast_1d(N_u)
    if N_u_array.size == 1:
        N_u_array = np.full_like(psi, float(N_u_array))
    elif N_u_array.size != psi.size:
        raise ValueError("N_u must be scalar or array of length N_psi")
    N_l_array = np.atleast_1d(N_l)
    if N_l_array.size == 1:
        N_l_array = np.full_like(psi, float(N_l_array))
    elif N_l_array.size != psi.size:
        raise ValueError("N_l must be scalar or array of length N_psi")

    # Build 3D mesh (vertical camber applied as mean-line z-offset in agreement with NACA convention)
    
    # Nondimensional y-coordinate
    eta = np.linspace(0.0, 1.0, N_eta)
    eta_clip = np.clip(eta, 1e-12, 1.0 - 1e-12)
    
    # Shape function (constant 2 for elliptic lobe)
    S_c = 2.0  # (24)
    S_u = S_c * np.ones_like(eta)  # (24)
    S_l = S_c * np.ones_like(eta)  # (24)

    # 2D arrays describing 3D mesh
    X = np.zeros((N_psi, N_eta))
    Y_U = np.zeros_like(X)
    Z_U = np.zeros_like(X)
    Y_L = np.zeros_like(X)
    Z_L = np.zeros_like(X)

    # Generate cross-section at every axial station
    for i, psi_i in enumerate(psi):
        
        w_half_i = w_half[i]
        h_half_i = h_half[i]
        dz_meanline_i = dz_meanline[i]   # vertical mean-line offset [m]
        N_u_i = N_u_array[i]
        N_l_i = N_l_array[i]
        
        # Class function
        C_u = eta_clip**N_u_i * (1 - eta_clip)**N_u_i  # (25)
        C_l = eta_clip**N_l_i * (1 - eta_clip)**N_l_i  # (25)
        
        # Nondimensional z-coordinate
        zeta_u = C_u * S_u  # (26)
        zeta_l = C_l * S_l  # (26)

        # Dimensional y-coordinate (from -w_half to +w_half)
        y = (eta - 0.5) * 2.0 * w_half_i  # translate to centreline
        Y_U[i, :] = y + dy
        Y_L[i, :] = y + dy
        
        # Dimensional z-coordinate (symmetric about centerline before camber)
        zeta_u_max = S_c * np.max(C_u) + 1e-12
        zeta_l_max = S_c * np.max(C_l) + 1e-12
        z_u = h_half_i * zeta_u / zeta_u_max
        z_l = -h_half_i * zeta_l / zeta_l_max
        # Apply vertical camber (shift upper and lower z by dz_meanline_i)
        Z_U[i, :] = dz_meanline_i + z_u + dz
        Z_L[i, :] = dz_meanline_i + z_l + dz

        # Dimensional x-coordinate
        X[i, :] = psi_i * length + dx

    # Output dictionary
    out_dict = {
        "x": X, "y_upper": Y_U, "z_upper": Z_U, "y_lower": Y_L, "z_lower": Z_L,
        "psi": psi, "eta": eta,
        "zeta_horz": zeta_horz, "zeta_vert": zeta_vert,
        "w_half": w_half, "h_half": h_half,
        "zeta_camber_vert": zeta_camber_vert,
        "dz_meanline": dz_meanline,
        "spinner_dict": spinner_dict,
    }

    if plot:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(out_dict['x'], out_dict['y_upper'], out_dict['z_upper'],
                        rstride=max(1, N_psi // 80), cstride=max(1, N_eta // 60),
                        linewidth=0, alpha=0.9)
        ax.plot_surface(out_dict['x'], out_dict['y_lower'], out_dict['z_lower'],
                        rstride=max(1, N_psi // 80), cstride=max(1, N_eta // 60),
                        linewidth=0, alpha=0.9)

        # plot spinner if present (attach at x=0)
        if spinner_dict is not None:
            # spinner x_mesh, y_mesh, z_mesh defined relative to spinner chord c_sp
            ax.plot_surface(spinner_dict["x_mesh"], spinner_dict["y_mesh"], spinner_dict["z_mesh"],
                            rstride=1, cstride=1, linewidth=0, alpha=0.85, color='gray')

        ax.set_xlabel('Axial x [m]')
        ax.set_ylabel('Lateral y [m]')
        ax.set_zlabel('Vertical z [m]')
        ax.set_title('Nacelle with optional camber (analytic)')
        ax.set_aspect('equal')
        ax.view_init(elev=22, azim=130)
        plt.tight_layout()
        plt.show()
        
    if write_dat_file:
        
        # 2) extract arrays
        X = np.asarray(out_dict["x"])            # shape (N_psi, N_eta)
        Y_u = np.asarray(out_dict["y_upper"])    # lateral (m)
        Z_u = np.asarray(out_dict["z_upper"])    # vertical (m)
        Y_l = np.asarray(out_dict["y_lower"])
        Z_l = np.asarray(out_dict["z_lower"])
        
        ns, nn = X.shape   # ns = spanwise (N_psi), nn = points per section (N_eta)
    
        # 3) create BasicSurface: choose n_sec=2 so there is exactly one "surface piece" slot per
        #    control-surface pair (n_sec-1), then set nn and ns to match array sizes.
        surf = BasicSurface(n_sec=2, name="eprop_nacelle", nn=nn, ns=ns)
    
        # 4) map coordinates to BasicSurface convention: [X, Y, Z]
        upper_piece = [X.copy(), Y_u.copy(), Z_u.copy()]
        lower_piece = [X.copy(), Y_l.copy(), Z_l.copy()]
    
        # 5) assign as two pieces (two Tecplot zones)
        surf.surfs = [upper_piece, lower_piece]
        
        # write nacelle
        surf.output_tecplot(fname="nacelle.dat", one_piece=False)
        
        # write spinner
        sp = out_dict.get("spinner_dict", None)
        if sp is not None:
        
            Xs = sp["x_mesh"]
            Ys = sp["y_mesh"]
            Zs = sp["z_mesh"]
        
            ns_s, nn_s = Xs.shape
        
            spinner = BasicSurface(n_sec=2, name="spinner", nn=nn_s, ns=ns_s)
            spinner.surfs = [[Xs, Ys, Zs]]
        
            spinner.output_tecplot(fname="spinner.dat", one_piece=False)

    return out_dict


def convert_to_watertight_surface_mesh(res, plot_mesh=False):

    # get arrays returned by your generator
    X = res["x"]            # shape (ns, nn)
    YU = res["y_upper"]
    ZU = res["z_upper"]
    YL = res["y_lower"]
    ZL = res["z_lower"]
    
    ns, nn = X.shape
    
    # --- build point list: stack upper then lower ---
    upper_pts = np.column_stack([X.ravel(order='C'), YU.ravel(order='C'), ZU.ravel(order='C')])
    lower_pts = np.column_stack([X.ravel(order='C'), YL.ravel(order='C'), ZL.ravel(order='C')])
    points = np.vstack([upper_pts, lower_pts])  # shape (2*ns*nn, 3)
    
    # helper index functions (row-major order: i = psi index, j = eta index)
    def idx_upper(i, j):
        return i * nn + j
    
    def idx_lower(i, j):
        return ns * nn + i * nn + j
    
    faces = []  # will contain [3, a, b, c] for each triangle
    
    # 1) Upper surface (triangulate each quad)
    for i in range(ns - 1):
        for j in range(nn - 1):
            a = idx_upper(i, j)
            b = idx_upper(i + 1, j)
            c = idx_upper(i + 1, j + 1)
            d = idx_upper(i, j + 1)
            faces.append([3, a, b, c])
            faces.append([3, a, c, d])
    
    # 2) Lower surface (triangulate each quad)
    # Note: orientation reversed relative to upper to try to keep outward normals consistent
    for i in range(ns - 1):
        for j in range(nn - 1):
            a = idx_lower(i, j)
            b = idx_lower(i, j + 1)
            c = idx_lower(i + 1, j + 1)
            d = idx_lower(i + 1, j)
            faces.append([3, a, b, c])
            faces.append([3, a, c, d])
    
    # 3) Side panels at lateral edges j=0 and j=nn-1 (connect upper and lower)
    # left side j=0
    j_edge = 0
    for i in range(ns - 1):
        a = idx_upper(i, j_edge)
        b = idx_upper(i + 1, j_edge)
        c = idx_lower(i + 1, j_edge)
        d = idx_lower(i, j_edge)
        faces.append([3, a, b, c])
        faces.append([3, a, c, d])
    
    # right side j=nn-1
    j_edge = nn - 1
    for i in range(ns - 1):
        a = idx_upper(i, j_edge)
        b = idx_lower(i, j_edge)
        c = idx_lower(i + 1, j_edge)
        d = idx_upper(i + 1, j_edge)
        faces.append([3, a, b, c])
        faces.append([3, a, c, d])
    
    # 4) Front cap at i=0 (fan triangulation about centroid)
    front_loop = [idx_upper(0, j) for j in range(nn)] + [idx_lower(0, j) for j in range(nn - 1, -1, -1)]
    front_pts = points[front_loop]
    front_centroid = front_pts.mean(axis=0)
    points = np.vstack([points, front_centroid])
    front_centroid_idx = points.shape[0] - 1
    for k in range(len(front_loop)):
        a = front_loop[k]
        b = front_loop[(k + 1) % len(front_loop)]
        faces.append([3, a, b, front_centroid_idx])
    
    # 5) Rear cap at i=ns-1
    rear_loop = [idx_upper(ns - 1, j) for j in range(nn)] + [idx_lower(ns - 1, j) for j in range(nn - 1, -1, -1)]
    rear_pts = points[rear_loop]
    rear_centroid = rear_pts.mean(axis=0)
    points = np.vstack([points, rear_centroid])
    rear_centroid_idx = points.shape[0] - 1
    for k in range(len(rear_loop)):
        a = rear_loop[k]
        b = rear_loop[(k + 1) % len(rear_loop)]
        faces.append([3, a, b, rear_centroid_idx])
    
    # convert faces to the flat format PyVista expects: [n0, v0, v1, v2, n1, ...] flattened
    faces_flat = np.hstack([np.array(f, dtype=np.int64) for f in faces])
    
    # create PolyData
    poly = pv.PolyData(points, faces_flat)
    
    # triangulate and clean (removes duplicate points, merge, etc.)
    poly = poly.triangulate().clean()
    
    print("Number of open edges (should be 0):", getattr(poly, "n_open_edges", "unknown"))
    print("Is all triangles?:", poly.is_all_triangles)
    
    if plot_mesh == True:
        
        # Optionally show the mesh to visually inspect
        p = pv.Plotter()
        p.add_mesh(poly, show_edges=True, opacity=1.0)
        p.show()
    
    # compute volume (requires a watertight closed triangulated surface)
    try:
        vol = poly.volume  # property that calls vtkMassProperties internally
        print("Closed-surface volume (m^3):", vol)
    except Exception as e:
        print("Error computing volume from triangle surface:", e)
        # fall back: try Delaunay 3D (tetrahedralization)
        ug = poly.delaunay_3d(alpha=0.0)  # tweak alpha if needed
        print("UnstructuredGrid total volume (m^3):", ug.volume)

    return poly

#%%

if __name__ == "__main__":
   
    # Inputs
    
    # Discretisation
    N_psi = 200
    N_eta = 200
    
    # Overall dimensions
    l_tot = 4.0
    w_tot = 1.2 * 0.8
    h_tot = 1.2 * 1.2

    # Leading edge
    w_le = 0.1 * 2
    h_le = 0.1 * 3
    
    # Trailing edge
    w_te = 0.15 * 2
    h_te = 0.1 * 3
    beta_horz = np.deg2rad(8.0)
    beta_vert = np.deg2rad(6.0)
    
    # Upper half
    N_u = 0.45
    
    # Lower half
    N_l = 0.25
    
    # Vertical camber
    camber_vert_tuple = (0.0, 0.35)  # (-0.01 * 4, 0.35)  # None
    
    res = generate_noflow_nacelle_geometry(
        l_tot, w_tot, h_tot,
        w_le, w_te, beta_horz,
        h_le, h_te, beta_vert,
        N_u=N_u, N_l=N_l,
        camber_vert_tuple=camber_vert_tuple,
        N_psi=N_psi, N_eta=N_eta,
        plot=False,
        write_dat_file=False,
    )
    
    # Convert open surface mesh to watertight surface mesh
    watertight_surface_mesh = convert_to_watertight_surface_mesh(res)
    
    #%%
    
    # nacelle_mesh = pv.read(r"C:\Users\nmb48\Documents\GitHub\cst-modeling3d\nacelle.dat")
    # spinner_mesh = pv.read(r"C:\Users\nmb48\Documents\GitHub\cst-modeling3d\spinner.dat")
    
    # plotter = pv.Plotter()
    # plotter.add_mesh(nacelle_mesh, show_edges=True)
    # plotter.add_mesh(spinner_mesh, show_edges=True)
    # plotter.show()
    
    plotter = pv.Plotter()
    plotter.add_mesh(watertight_surface_mesh, show_edges=True)
    plotter.show()
        
    #%%
    
    from nils.bin_packing.shared_functions import pv_to_trimesh
    tm = pv_to_trimesh(watertight_surface_mesh)
    
    print('tm.is_watertight =', tm.is_watertight)
        
        
    
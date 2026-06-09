__all__ = [
    "build_closed_poly",
    "is_closed_surface",
    "build_camber_profile",
]

import numpy as np
import pyvista as pv


def build_closed_poly(meshes, merge_points=False, hole_size=1e6):
    """Merge a sequence of surface.surfs indices into a closed triangular PolyData.
    - surface: your BasicSurface instance
    - merge_points: pass to pv.merge to attempt point merging (False safer)
    - hole_size: passed to fill_holes to cap openings (adjust)
    """
    # Merge sub-meshes
    merged = pv.merge(meshes, merge_points=merge_points)
    # Get surface, triangulate, and clean/remove duplicates
    poly = merged.extract_surface().triangulate().clean()
    # Fill holes (large hole_size ensures end-caps get filled)
    poly = poly.fill_holes(hole_size)
    # Recompute normals and orient consistently
    poly.compute_normals(auto_orient_normals=True, inplace=True)
    return poly


def is_closed_surface(mesh: pv.StructuredGrid) -> bool:
    # Convert to polygonal surface
    poly = mesh.extract_surface().triangulate()
    # Extract edges belonging to only one face
    open_edges = poly.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    return open_edges.n_cells == 0


def _quadratic(x0, y0, x1, y1):
    a = (y1 - y0) / (x1 - x0) ** 2
    b = -2.0 * a * x0
    c = y0 + a * x0 ** 2
    return np.poly1d([a, b, c])


def _class_function(psi, N1=0.5, N2=1.0):
    psi = np.clip(np.asarray(psi), 1e-12, 1.0 - 1e-12)
    return psi ** N1 * (1.0 - psi) ** N2


def build_camber_profile(psi, camber_tuple, envelope, N1, N2, N_psi):
    """
    Minimal camber builder copied from noflow_nacelle_geometry_generator.
    camber_tuple = (camber_max, psi_cam)
    Returns psi, camber_nondim (so physical mean-line m(psi)*envelope in meters).
    """
    camber_max_in, psi_cam = camber_tuple if camber_tuple is not None else (0.0, 0.5)
    camber_max = camber_max_in

    camber_te = 0.0
    S_le = 0.0
    S_te = 0.0

    Psi_c = float(np.clip(psi_cam, 1e-4, 1.0 - 1e-4))
    S_camber_max = (camber_max - Psi_c * camber_te) / (np.sqrt(Psi_c) * (1.0 - Psi_c) + 1e-12)

    quad_front = _quadratic(Psi_c, S_camber_max, 0.0, S_le)
    quad_aft = _quadratic(Psi_c, S_camber_max, 1.0, S_te)

    S_list = np.where(psi >= Psi_c, quad_aft(psi), quad_front(psi))
    C = _class_function(psi, N1=N1, N2=N2)
    camber = C * S_list + psi * camber_te
    camber[0] = 0.0
    camber[-1] = camber_te
    return psi, camber
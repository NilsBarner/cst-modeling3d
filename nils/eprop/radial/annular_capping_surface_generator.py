""""""

__all__ = [
    "generate_annular_capping_surface",
    "end_ring_points",
]

import numpy as np
import pyvista as pv


# ------------------------------------------------------------
# Generate two concentric rings with different resolutions
# ------------------------------------------------------------
def make_ring(radius, x_offset, n_points):
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)

    y = radius * np.cos(theta)
    z = radius * np.sin(theta)
    x = np.full_like(y, x_offset)

    return np.column_stack([x, y, z])


# ------------------------------------------------------------
# Resample a ring to N points (arc-length parametrised)
# ------------------------------------------------------------
def resample_loop(points, n):
    pts = np.vstack([points, points[0]])  # close loop

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    s /= s[-1]

    t = np.linspace(0, 1, n, endpoint=False)

    new_pts = np.zeros((n, 3))
    for i in range(3):
        new_pts[:, i] = np.interp(t, s, pts[:, i])

    return new_pts


# ------------------------------------------------------------
# Build annular surface (the important part)
# ------------------------------------------------------------
def generate_annular_capping_surface(outer_pts, inner_pts, n=200):
    outer = resample_loop(outer_pts, n)
    inner = resample_loop(inner_pts, n)

    pts = np.vstack([outer, inner])

    faces = []

    for i in range(n):
        i_next = (i + 1) % n  # <-- critical for closure

        o0 = i
        o1 = i_next
        i0 = i + n
        i1 = i_next + n

        # two triangles per quad
        faces.extend([3, o0, o1, i1])
        faces.extend([3, o0, i1, i0])

    faces = np.array(faces)

    surf = pv.PolyData(pts, faces).clean()
    return surf


def end_ring_points(mesh, which_end="min", atol=None):
    """
    Extract boundary ring points near x=min or x=max from a surface mesh.
    Assumes the desired ring is approximately planar at constant x.
    """
    surf = mesh.extract_surface().triangulate().clean()
    pts = np.asarray(surf.points)

    x_end = pts[:, 0].min() if which_end == "min" else pts[:, 0].max()

    if atol is None:
        b = np.array(surf.bounds, dtype=float)
        diag = np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
        atol = 1e-8 * diag

    ring = pts[np.isclose(pts[:, 0], x_end, atol=atol)]
    ring = np.unique(np.round(ring, 12), axis=0)

    if ring.shape[0] < 3:
        raise ValueError("Could not extract a valid ring at the requested end.")

    # sort by angle in the yz-plane
    c = ring.mean(axis=0)
    ang = np.mod(np.arctan2(ring[:, 2] - c[2], ring[:, 1] - c[1]), 2.0 * np.pi)
    ring = ring[np.argsort(ang)]

    return ring

#%%

if __name__ == '__main__':
    
    outer_pts = make_ring(radius=1.0, x_offset=0.0,  n_points=137)
    inner_pts = make_ring(radius=0.5, x_offset=0.15, n_points=83)
    bridge = generate_annular_capping_surface(outer_pts, inner_pts, n=300)

    # ------------------------------------------------------------
    # Watertightness check (for the annular surface alone)
    # ------------------------------------------------------------
    def check_closed(mesh):
        edges = mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        return edges.n_points == 0
    
    
    print("Bridge watertight (no boundary edges):", check_closed(bridge))
    
    
    # ------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------
    plotter = pv.Plotter()
    plotter.add_mesh(bridge, color="red", show_edges=True)
    
    # show original rings for reference
    plotter.add_points(outer_pts, color="black", point_size=6)
    plotter.add_points(inner_pts, color="blue", point_size=6)
    
    plotter.show()
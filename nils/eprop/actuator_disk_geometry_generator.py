"""
Minimal script to generate an annular actuator-disk surface in the plane x = x_plane
and return it as a pyvista.StructuredGrid suitable for plotting with Plotter.add_mesh().
"""
__all__ = ["generate_actuator_disk_geometry"]

import numpy as np
import pyvista as pv
from typing import Union


def generate_actuator_disk_geometry(r_out: float, r_in: float, x_plane: float, y_plane: float, z_plane: float) -> pv.StructuredGrid:
    """
    Generate an annular surface (disk with hole) lying in the plane x = x_plane.

    Parameters
    ----------
    r_out : float
        Outer radius of the annulus (must be > 0).
    r_in : float
        Inner radius of the annulus (0 <= r_in < r_out).
    x_plane : float
        The x coordinate where the entire annulus lives (constant).

    Returns
    -------
    pyvista.StructuredGrid
        A structured-grid representing the annular surface (shape: (n_radial, n_theta, 1)).
    """
    if not (0 <= r_in < r_out):
        raise ValueError("Require 0 <= r_in < r_out")

    # sampling resolution (tweak if you want finer/coarser mesh)
    n_radial = 32      # number of samples in the radial direction
    n_theta = 128      # number of samples around the circumference

    # radial samples from inner to outer radius
    radial = np.linspace(r_in, r_out, n_radial)
    # angular samples; endpoint=False to avoid duplicating theta==0 and theta==2pi
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=True)

    # create 2D grid in (r,theta) with indexing='ij' -> shapes (n_radial, n_theta)
    R, TH = np.meshgrid(radial, theta, indexing='ij')

    # convert polar (R,TH) -> Cartesian in the plane x = x_plane
    X = np.full_like(R, fill_value=float(x_plane))
    Y = R * np.cos(TH) + y_plane
    Z = R * np.sin(TH) + z_plane

    # expand to 3D arrays with a singleton third dimension so we have (n_radial, n_theta, 1)
    X3 = X[:, :, np.newaxis]
    Y3 = Y[:, :, np.newaxis]
    Z3 = Z[:, :, np.newaxis]

    # build and return a StructuredGrid
    grid = pv.StructuredGrid(X3, Y3, Z3)
    return grid

#%%

if __name__ == "__main__":
    # quick test / demonstration
    r_outer = 1.0
    r_inner = 0.25
    x_loc = 0.0

    grid = generate_actuator_disk_geometry(r_outer, r_inner, x_loc)

    # create a Plotter and add the mesh; this will open an interactive window
    p = pv.Plotter(window_size=(800, 600))
    p.add_mesh(grid, show_edges=False, color='lightgrey', opacity=0.5, smooth_shading=True)
    p.add_axes()         # optional: show axes for orientation
    p.add_scalar_bar(title="dummy")  # harmless; structured grid has no scalars but doesn't break anything
    p.show(title="Actuator Disk (annular surface)")
    
    
    
"""
This script parametrises and plots the an integrated inclined ducted radiator.
It uses docs\example\jet-engine.py to loft a 3D surface over several 2D
cross-sections. nils\ducted_radiator\inclined\duct_geometry_getter.py is used
to create the variable-geometry, curved upstream and downstream duct merging
into and originating from the inclined heat exchanger.
"""

__all__ = ["generate_inclined_ducted_radiator_geometry"]

import sys
import math
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt

from cst_modeling.basic import BasicSection, BasicSurface
from cst_modeling.section import cst_curve
from nils.eprop.inclined.duct_geometry_getter import get_duct_geometry
from nils.eprop.shared_functions import Section, curve, order_curve3d


def generate_inclined_ducted_radiator_geometry(
    # Intake
    dz_hx_intake,
    w_intake,
    h_intake,
    # Upstream duct
    l_up_duct,
    # HX
    l_hx,
    w_hx,
    h_hx,
    dx_hx_corner,
    dz_hx_corner,
    alpha,
    beta,
    # Downstream duct
    l_down_duct,
    # Fan
    dz_hx_fan,
    d_fan,
    l_fan,
    # Nozzle
    d_nozzle,
    l_nozzle,
    
    N_inlet_up,
    N_outlet_up,
    n_slices_up,
    
    N_inlet_down,
    N_outlet_down,
    n_slices_down,
    
    n_surf_sec,
    nn_sect,
    nn_surf,
    ns_surf,
    
    plot_sections=False,
    plot_scatter=False,
    plot_mesh=False,
    save_dat=False,
    
    dx=0.0,
    dy=0.0,
    dz=0.0,
):
    
    x_centroid_hx = 0.0
    y_centroid_hx = 0.0
    z_centroid_hx = 0.0
    
    # Instantiate BasicSurface object
    surface = BasicSurface(n_sec=n_surf_sec, name='out', nn=nn_surf, ns=ns_surf, projection=False)
    
    ### NILS: upstream duct
    
    i_up_duct_start = 0
    i_up_duct_end = n_slices_up - 1
    
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
    
    angle_outlet = -(90 - alpha)
    # angle_outlet = -(180 - (alpha + beta))
    
    include_endpoints = False  # True
    is_inlet = True
    
    slices_list = get_duct_geometry(
        x_centroid_inlet, y_centroid_inlet, z_centroid_inlet,
        x_centroid_outlet, y_centroid_outlet, z_centroid_outlet,
        w_inlet, h_inlet,
        w_outlet, h_outlet,
        angle_inlet, angle_outlet,
        N_inlet_up,
        N_outlet_up,
        n_slices_up,
        include_endpoints,
        is_inlet,
        side_slope_start_deg=0.0,
        side_slope_end_deg=180 - (alpha + beta),
        match_slopes=False,
        plot_sections=plot_sections,
    )
    
    for i, _slice in enumerate(slices_list):
        
        # Convert into `Section` in context of this script
        surface.secs[i] = Section(n_curve = 1)
        surface.secs[i].curve[0].curve_x = _slice['Z']
        surface.secs[i].curve[0].curve_y = _slice['Y']
        surface.secs[i].curve[0].curve_z = _slice['X']
        surface.secs[i].join_curve()
    
    ### NILS: HX inlet
    
    i = n_slices_up
    
    surface.secs[i] = Section(n_curve = 4)
    surface.secs[i].curve[0] = curve(
        r0=np.array([z_centroid_hx - h_hx/2, y_centroid_hx + w_hx/2, x_centroid_hx + dx_hx_corner]),
        r1=np.array([z_centroid_hx + dz_hx_corner, y_centroid_hx + w_hx/2, x_centroid_hx - l_hx/2]),
        nn=3,  # to match 1 point on horizontal symmetry axis as obtained from CST parametrisation with low N exponent
    )
    surface.secs[i].curve[1] = curve(
        r0=np.array([z_centroid_hx + dz_hx_corner, y_centroid_hx + w_hx/2, x_centroid_hx - l_hx/2]),
        r1=np.array([z_centroid_hx + dz_hx_corner, y_centroid_hx - w_hx/2, x_centroid_hx - l_hx/2]),
        nn=nn_sect//2-3,
    )
    surface.secs[i].curve[2] = curve(
        r0=np.array([z_centroid_hx + dz_hx_corner, y_centroid_hx - w_hx/2, x_centroid_hx - l_hx/2]),
        r1=np.array([z_centroid_hx - h_hx/2, y_centroid_hx - w_hx/2, x_centroid_hx + dx_hx_corner]),
        nn=3,
    )
    surface.secs[i].curve[3] = curve(
        r0=np.array([z_centroid_hx - h_hx/2, y_centroid_hx - w_hx/2, x_centroid_hx + dx_hx_corner]),
        r1=np.array([z_centroid_hx - h_hx/2, y_centroid_hx + w_hx/2, x_centroid_hx + dx_hx_corner]),
        nn=nn_sect//2-3,
    )
    surface.secs[i].join_curve()
    
    ### NILS: HX outlet
    
    i = i + 1
    
    surface.secs[i] = Section(n_curve = 4)
    surface.secs[i].curve[0] = curve(
        r0=np.array([z_centroid_hx - dz_hx_corner, y_centroid_hx + w_hx/2, x_centroid_hx + l_hx/2]),
        r1=np.array([z_centroid_hx + h_hx/2, y_centroid_hx + w_hx/2, x_centroid_hx - dx_hx_corner]),
        nn=3,  # to match 1 point on horizontal symmetry axis as obtained from CST parametrisation with low N exponent
    )
    surface.secs[i].curve[1] = curve(
        r0=np.array([z_centroid_hx + h_hx/2, y_centroid_hx + w_hx/2, x_centroid_hx - dx_hx_corner]),
        r1=np.array([z_centroid_hx + h_hx/2, y_centroid_hx - w_hx/2, x_centroid_hx - dx_hx_corner]),
        nn=nn_sect//2-3,
    )
    surface.secs[i].curve[2] = curve(
        r0=np.array([z_centroid_hx + h_hx/2, y_centroid_hx - w_hx/2, x_centroid_hx - dx_hx_corner]),
        r1=np.array([z_centroid_hx - dz_hx_corner, y_centroid_hx - w_hx/2, x_centroid_hx + l_hx/2]),
        nn=3,
    )
    surface.secs[i].curve[3] = curve(
        r0=np.array([z_centroid_hx - dz_hx_corner, y_centroid_hx - w_hx/2, x_centroid_hx + l_hx/2]),
        r1=np.array([z_centroid_hx - dz_hx_corner, y_centroid_hx + w_hx/2, x_centroid_hx + l_hx/2]),
        nn=nn_sect//2-3,
    )
    surface.secs[i].join_curve()
    
    ### NILS: downstream duct
    
    _i = i + 1
    
    i_down_duct_start = _i
    i_down_duct_end = i_down_duct_start + n_slices_down - 1
    
    x_centroid_inlet = x_centroid_hx + (l_hx/2 - dx_hx_corner) / 2
    y_centroid_inlet = y_centroid_hx
    z_centroid_inlet = z_centroid_hx + (h_hx/2 - dz_hx_corner) / 2
    x_centroid_outlet = x_centroid_hx + l_hx/2 + l_down_duct
    y_centroid_outlet = y_centroid_hx
    z_centroid_outlet = z_centroid_hx + dz_hx_fan
    w_inlet = w_hx
    h_inlet = (h_hx/2 + dz_hx_corner)# / np.cos(np.deg2rad(90 - 22.78))
    w_outlet = d_fan
    h_outlet = d_fan

    angle_inlet = -(90 - alpha)
    # angle_inlet = -(180 - (alpha + beta))
    
    angle_outlet = 0.0
    include_endpoints = False  # True
    is_inlet = False
    
    slices_list = get_duct_geometry(
        x_centroid_inlet, y_centroid_inlet, z_centroid_inlet,
        x_centroid_outlet, y_centroid_outlet, z_centroid_outlet,
        w_inlet, h_inlet,
        w_outlet, h_outlet,
        angle_inlet, angle_outlet,
        N_inlet_down,
        N_outlet_down,
        n_slices_down,
        include_endpoints,
        is_inlet,
        side_slope_start_deg=180 - (alpha + beta),
        side_slope_end_deg=0.0,
        match_slopes=False,
        plot_sections=plot_sections,
    )
    
    for j, _slice in enumerate(slices_list):
        
        i = j + _i
        
        # Convert into `Section` in context of this script
        surface.secs[i] = Section(n_curve = 1)
        surface.secs[i].curve[0].curve_x = _slice['Z']
        surface.secs[i].curve[0].curve_y = _slice['Y']
        surface.secs[i].curve[0].curve_z = _slice['X']
        surface.secs[i].join_curve()
    
    ### NILS: fan inlet
    
    i = i + 1
    
    x_centroid_fan_inlet = x_centroid_hx + l_hx/2 + l_down_duct
    y_centroid_fan_inlet = z_centroid_hx + dz_hx_fan  # NILS: note coordinate change
    z_centroid_fan_inlet = y_centroid_hx
    
    surface.secs[i] = Section(n_curve = 4)
    surface.secs[i].circle(
        np.array([y_centroid_fan_inlet - d_fan/2, z_centroid_fan_inlet]),
        np.array([y_centroid_fan_inlet + d_fan/2, z_centroid_fan_inlet]),
        np.array([y_centroid_fan_inlet, z_centroid_fan_inlet + d_fan/2]),
        nn=nn_sect,
    )
    surface.secs[i].z = x_centroid_fan_inlet * np.ones_like(surface.secs[i].x)
    
    surface.secs[i].x, surface.secs[i].y, surface.secs[i].z = order_curve3d(
        surface.secs[i].x,
        surface.secs[i].y,
        surface.secs[i].z,
    )
    
    surface.secs[i].x[0] = surface.secs[i].x[-1]
    surface.secs[i].y[0] = surface.secs[i].y[-1]
    surface.secs[i].z[0] = surface.secs[i].z[-1]
    
    ### NILS: fan outlet
    
    i = i + 1
    
    x_centroid_fan_outlet = x_centroid_hx + l_hx/2 + l_down_duct + l_fan
    y_centroid_fan_outlet = y_centroid_hx + dz_hx_fan  # NILS: note coordinate change
    z_centroid_fan_outlet = z_centroid_hx
    
    surface.secs[i] = Section(n_curve = 4)
    surface.secs[i].circle(
        np.array([y_centroid_fan_outlet - d_fan/2, z_centroid_fan_outlet]),
        np.array([y_centroid_fan_outlet + d_fan/2, z_centroid_fan_outlet]),
        np.array([y_centroid_fan_outlet, z_centroid_fan_outlet + d_fan/2]),
        nn=nn_sect,
    )
    surface.secs[i].z = x_centroid_fan_outlet * np.ones_like(surface.secs[i].x)
    
    surface.secs[i].x, surface.secs[i].y, surface.secs[i].z = order_curve3d(
        surface.secs[i].x,
        surface.secs[i].y,
        surface.secs[i].z,
    )
    
    surface.secs[i].x[0] = surface.secs[i].x[-1]
    surface.secs[i].y[0] = surface.secs[i].y[-1]
    surface.secs[i].z[0] = surface.secs[i].z[-1]
    
    ### NILS: nozzle outlet
    
    i = i + 1
    
    x_centroid_nozzle_outlet = x_centroid_hx + l_hx/2 + l_down_duct + l_fan + l_nozzle
    y_centroid_nozzle_outlet = y_centroid_hx + dz_hx_fan  # NILS: note coordinate change
    z_centroid_nozzle_outlet = z_centroid_hx
    
    surface.secs[i] = Section(n_curve = 4)
    surface.secs[i].circle(
        np.array([y_centroid_nozzle_outlet - d_nozzle/2, z_centroid_nozzle_outlet]),
        np.array([y_centroid_nozzle_outlet + d_nozzle/2, z_centroid_nozzle_outlet]),
        np.array([y_centroid_nozzle_outlet, z_centroid_nozzle_outlet + d_nozzle/2]),
        nn=nn_sect,
    )
    surface.secs[i].z = x_centroid_nozzle_outlet * np.ones_like(surface.secs[i].x)
    
    surface.secs[i].x, surface.secs[i].y, surface.secs[i].z = order_curve3d(
        surface.secs[i].x,
        surface.secs[i].y,
        surface.secs[i].z,
    )
    
    surface.secs[i].x[0] = surface.secs[i].x[-1]
    surface.secs[i].y[0] = surface.secs[i].y[-1]
    surface.secs[i].z[0] = surface.secs[i].z[-1]
    
    # Scatter plot of 3D cross-sections
    
    if plot_scatter == True:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for i in range(len(surface.secs)):
            ax.scatter(surface.secs[i].x + dx, surface.secs[i].y + dy, surface.secs[i].z + dz)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect('equal')
        plt.show()
        
    # =============================================================================
    # Translate mesh by dx, dy, and dz
    # for surface in [outer_duct_surface, inner_duct_surface, outer_hx_surface, inner_hx_surface]:
    for i in range(len(surface.secs)):
        surface.secs[i].x += dz  # NILS: note flipped coordinates
        surface.secs[i].y += dy
        surface.secs[i].z += dx  # NILS: note flipped coordinates
    # =============================================================================
    
    # Convert sections to mesh
    
    surface.geo(update_sec=False)
    
    # surface.smooth(i_up_duct_start, i_up_duct_end, smooth0=True, ratio_end=-1)
    # surface.smooth(i_down_duct_start, i_down_duct_end, smooth0=False, ratio_end=-1)  # NILS: set `smooth0` to False, else get ugly ripples!
    
    surface.flip(axis = '+Y')
    surface.flip(axis = '+X')
    
    # Create single master mesh
    
    # def surface_to_vtk(surface):
    #     meshes = [pv.StructuredGrid(*surf) for surf in surface.surfs]
    #     return pv.merge(meshes)
    
    # master_mesh = surface_to_vtk(surface)
    
    # =============================================================================
    def surface_to_vtk_with_regions(surface,
                                    i_up_duct_start, i_up_duct_end,
                                    n_slices_up,
                                    i_down_duct_start, i_down_duct_end):
        """
        Convert surface.surfs -> merged StructuredGrid and add integer 'region' point-data.
        Regions (example mapping):
          0 : upstream duct (i_up_duct_start .. i_up_duct_end)
          1 : HX inlet          (index == n_slices_up)
          2 : HX outlet         (index == n_slices_up + 1)
          3 : downstream duct   (i_down_duct_start .. i_down_duct_end)
          4 : fan / nozzle / remainder
        """
        meshes = []
        for idx, surf in enumerate(surface.surfs):
            mesh = pv.StructuredGrid(*surf)   # surf = [surf_x, surf_y, surf_z] (ns, nn)
            # determine region id
            if i_up_duct_start <= idx <= i_up_duct_end:
                rid = 0
            elif i_up_duct_end <= idx <= i_up_duct_end + 1:
                rid = 1
            elif i_up_duct_end + 1 <= idx <= i_down_duct_end:
                rid = 2
            elif i_down_duct_end <= idx <= i_down_duct_end + 1:
                rid = 3
            else:
                rid = 4
            # attach integer point-wise region id
            mesh.cell_data['region'] = np.full(mesh.n_cells, rid, dtype=np.int32)
            meshes.append(mesh)
    
        # merge into single VTK dataset (region array is preserved)
        master = pv.merge(meshes)
        return master
    
    
    # ... after you have created `surface` and computed i_up_duct_start, i_up_duct_end, etc.
    master_mesh = surface_to_vtk_with_regions(
        surface,
        i_up_duct_start, i_up_duct_end,
        n_slices_up,
        i_down_duct_start, i_down_duct_end,
    )
    # =============================================================================
    
    # Flip y- and z-axes to match convention
    master_mesh_flipped = master_mesh.copy()
    master_mesh_flipped.points[:, [1, 2]] = master_mesh_flipped.points[:, [2, 1]]
    
    if plot_mesh == True:
        
        # # Clip volume at xz-plane
        # plotter.add_volume_clip_plane(master_mesh_flipped, normal='-y')#, opacity=opacity[::-1], cmap='magma')
        
        plotter = pv.Plotter()
        plotter.add_mesh(master_mesh_flipped, show_edges=True)
        plotter.show()
        
        # # =============================================================================
        # # Define a categorical colormap
        # region_colors = ['red', 'green', 'blue', 'orange', 'purple']  # one color per region ID
        
        # plotter = pv.Plotter()
        # plotter.add_mesh(
        #     master_mesh_flipped,
        #     scalars='region',            # use the 'region' array to color
        #     # categorical=True,            # treat scalars as categories
        #     show_edges=False,             # optional: show mesh edges
        #     clim=[0, 4],                 # optional: ensure color map covers all region IDs
        #     cmap=region_colors,
        #     opacity=1.0,
        # )
        # plotter.add_legend([('Upstream duct', 'red'),
        #                     ('HX inlet', 'green'),
        #                     ('HX outlet', 'blue'),
        #                     ('Downstream duct', 'orange'),
        #                     ('Fan/Nozzle', 'purple')])
        # plotter.show()
        # # =============================================================================
        
    if save_dat == True:
       surface.output_tecplot(fname='inclined_ducted_radiator_mesh.dat')  # NILS
    
    # if plot_mesh == True:
    #     master_mesh_flipped = pv.read('inclined_ducted_radiator_mesh.dat')
    #     master_mesh_flipped.plot(border=True, border_color='k')
    
    return master_mesh_flipped

#%%

if __name__== "__main__":
    
    # Intake
    dz_hx_intake = -0.5
    w_intake = 0.75
    h_intake = 0.25
    # Upstream duct
    l_up_duct = 3
    # HX
    l_hx = 1.5
    w_hx = 1
    h_hx = 0.75
    dx_hx_corner = 0.5
    dz_hx_corner = 0.15
    alpha = 22.78
    beta = 115.23
    # Downstream duct
    l_down_duct = 2
    # Fan
    dz_hx_fan = 0.5
    d_fan = 0.5
    l_fan = 0.3
    # Nozzle
    d_nozzle = 0.3
    l_nozzle = 0.3
    
    N_inlet_up = 0.25
    N_outlet_up = 0.005
    n_slices_up = 10
    
    N_inlet_down = 0.005
    N_outlet_down = 0.5
    n_slices_down = 10
    
    n_surf_sec = n_slices_up + n_slices_down + 5
    nn_sect = 500
    nn_surf = 201
    ns_surf = 51
    
    plot_sections = False
    plot_scatter = False
    plot_mesh = True
    save_dat = False
    
    dx = 0.0
    dy = 0.0
    dz = 0.0
    
    inclined_ducted_radiator_mesh = generate_inclined_ducted_radiator_geometry(
        # Intake
        dz_hx_intake=dz_hx_intake,
        w_intake=w_intake,
        h_intake=h_intake,
        # Upstream duct
        l_up_duct=l_up_duct,
        # HX
        l_hx=l_hx,
        w_hx=w_hx,
        h_hx=h_hx,
        dx_hx_corner=dx_hx_corner,
        dz_hx_corner=dz_hx_corner,
        alpha=alpha,
        beta=beta,
        # Downstream duct
        l_down_duct=l_down_duct,
        # Fan
        dz_hx_fan=dz_hx_fan,
        d_fan=d_fan,
        l_fan=l_fan,
        # Nozzle
        d_nozzle=d_nozzle,
        l_nozzle=l_nozzle,
        
        N_inlet_up=N_inlet_up,
        N_outlet_up=N_outlet_up,
        n_slices_up=n_slices_up,
        
        N_inlet_down=N_inlet_down,
        N_outlet_down=N_outlet_down,
        n_slices_down=n_slices_down,
        
        n_surf_sec=n_surf_sec,
        nn_sect=nn_sect,
        nn_surf=nn_surf,
        ns_surf=ns_surf,
        
        plot_sections=plot_sections,
        plot_scatter=plot_scatter,
        plot_mesh=plot_mesh,
        save_dat=save_dat,
        
        dx=dx,
        dy=dy,
        dz=dz,
    )
    
    #%%
    
    # =============================================================================
    from nils.bin_packing.shared_functions import convert_core_mesh_to_watertight_surface_mesh
    watertight_surface_mesh = convert_core_mesh_to_watertight_surface_mesh(inclined_ducted_radiator_mesh)
    # =============================================================================
    
    from nils.bin_packing.shared_functions import pv_to_trimesh
    tm = pv_to_trimesh(watertight_surface_mesh)
    
    print('tm.is_watertight =', tm.is_watertight)
    
    #%%
    
    plotter = pv.Plotter()
    plotter.add_mesh(watertight_surface_mesh)
    plotter.show()
    

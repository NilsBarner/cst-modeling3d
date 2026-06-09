"""
This script plots the 3D parametric geometry of
the parallel electric ducted fan design, to which
separate heat exchanger nacelles will be attached.
"""

__all__ = ["generate_parallel_edf_geometry"]

import os
import sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from cst_modeling.io import output_surface
from cst_modeling.operation import Lofting_Revolution
from cst_modeling.tools.nils.parallel_edf_profile import PoweredNacelleProfile
from nils.edf.shared_functions import build_closed_poly, is_closed_surface


def generate_parallel_edf_geometry(
    n_point_segment,
    r_hx_out,
    r_hx_in,
    r_hub_rotor,
    r_tip_rotor,
    l_rotor,
    l_stator,
    h_stator_in,
    h_stator_out,
    l_hx_side,
    alpha_incl,
    l_bp_nozzle,
    r_bp_nozzle_out,
    l_core_nozzle,
    r_core_nozzle_out,
    beta_core_nozzle,
    beta_bp_nozzle,
    l_spinner,
    l_intake,
    bypass_inner_angle,
    n_spanwise,
    save_dat=False,
    plot_profile=False,
    plot_mesh=False,
    f_scale=1.0,
):

    circum_control_psi = [0.0, 90.0, 180.0, 270.0]
    
    # 2D profile
    profiles = []
    psi = circum_control_psi[0]
    nacelle_profile = PoweredNacelleProfile(psi=psi, n_point_segment=n_point_segment)
    nacelle_profile.set_parameters(
        r_hx_out=r_hx_out,
        r_hx_in=r_hx_in,
        r_hub_rotor=r_hub_rotor,
        r_tip_rotor=r_tip_rotor,
        l_rotor=l_rotor,
        l_stator=l_stator,
        h_stator_in=h_stator_in,
        h_stator_out=h_stator_out,
        l_hx_side=l_hx_side,
        alpha_incl=alpha_incl,
        l_bp_nozzle=l_bp_nozzle,
        r_bp_nozzle_out=r_bp_nozzle_out,
        l_core_nozzle=l_core_nozzle,
        r_core_nozzle_out=r_core_nozzle_out,
        beta_core_nozzle=beta_core_nozzle,
        beta_bp_nozzle=beta_bp_nozzle,
        l_spinner=l_spinner,
        l_intake=l_intake,
        cst_u=np.array([ 0.10, 0.10, 0.10, 0.10, 0.15]) * 2, # NILS: added factor 2
        cst_l=[-0.10, 0.15,-0.10, 0.05, 0.05],
        bypass_inner_angle=bypass_inner_angle,
        bypass_inner_control_points=[],
        core_outer_control_points=[],
        core_inner_control_points=[],
        f_scale=f_scale,
    )
    profile_x, profile_y = nacelle_profile.get_profile()
    # profiles.append([np.concatenate(profile_x), np.concatenate(profile_y)])  # cannot concatenate into one single array, as sub-arrays not ordered, resulting in weird connecting lines (see below plots)
    profiles.append([profile_x, profile_y])
    
    # Plot 2D profile(s)
    
    if plot_profile == True:
        nacelle_profile.plot(show=True)
        # fig, ax = plt.subplots()
        # ax.plot(np.concatenate(profile_x), np.concatenate(profile_y))
        # plt.show()
        # # vs
        # fig, ax = plt.subplots()
        # for i in range(len(profile_x)):
        #     ax.plot(profile_x[i], profile_y[i])
        # plt.show()
        
    # 3D axisymmetric body of revolution
    
    surfs_list = []
    section_s_loc=[0.0, 0.25, 0.50, 0.75]
    for [profile_x, profile_y] in profiles:
        for sub_profile_x, sub_profile_y in zip(profile_x, profile_y):
            
            # Prepare inputs to `Lofting_Revolution`
            
            radial_abs = sub_profile_y.copy()
            axial_abs = sub_profile_x.copy()
            
            axial_rel = axial_abs - axial_abs[0]
            radial_rel = radial_abs - radial_abs[0]
            
            unit_profile = [axial_rel, radial_rel]
            sub_profiles = [unit_profile, unit_profile, unit_profile, unit_profile]
            
            nacelle = Lofting_Revolution(
                profiles=sub_profiles,
                section_s_loc=section_s_loc,
                section_x=axial_abs[0],
                section_radius=radial_abs[0],
                section_scale=1.0,
                n_spanwise=n_spanwise,
            )
            
            # # Plot individual sub-profiles
            # fig, ax = plt.subplots()
            # ax.plot(sub_profile_x, sub_profile_y)
            # plt.show()
            
            surfs = nacelle.sweep(interp_profile_kind='linear')
            surfs_list.append(surfs)
    
        if plot_mesh == True:
            plotter = pv.Plotter()
    
        all_meshes = []
    
        # Create .dat file from surface generated from each revolved sub-profile
        for i, surfs in enumerate(surfs_list):
            for i_surf, surf in enumerate(surfs):
                
                if save_dat == True:
                    path = os.path.dirname(sys.argv[0])
                    output_surface(surf, fname=os.path.join(path, f'parallel_edf_mesh_{i}.dat'), ID=i_surf)
                
                mesh = pv.StructuredGrid(*surf)  # create VTK dataset from surface coordinates
                
                # store mesh
                all_meshes.append(mesh)
                
                if plot_mesh == True:
                    plotter.add_mesh(mesh, show_edges=True)                
                
                
        master_mesh = pv.merge(all_meshes)
        
        # Create separate surfaces for nacelle cross-sections and fan
        
        # fig, ax = plt.subplots()
        # ax.plot(nacelle_profile.profile_segments[4][:, 0], nacelle_profile.profile_segments[4][:, 1])
        # ax.plot(nacelle_profile.profile_segments[5][:, 0], nacelle_profile.profile_segments[5][:, 1])
        # ax.plot(nacelle_profile.profile_segments[6][:, 0], nacelle_profile.profile_segments[6][:, 1])
        # plt.show()
        
        # Define solid-body surfaces
        outer_surfs_list = [
            surfs_list[nacelle_profile.profile_mapping[4]],
            surfs_list[nacelle_profile.profile_mapping[5]],
            surfs_list[nacelle_profile.profile_mapping[6]],
        ]
        core_surfs_list = [
            surfs_list[nacelle_profile.profile_mapping[0]],
        ]
        
        # Convert surfaces to meshes
        outer_meshes = list(np.concatenate([
            [pv.StructuredGrid(*surf) for surf in outer_surfs] for outer_surfs in outer_surfs_list
        ]))
        core_meshes = list(np.concatenate([
            [pv.StructuredGrid(*surf) for surf in core_surfs] for core_surfs in core_surfs_list
        ]))
        
        # Build closed polys for each logical body
        outer_poly = build_closed_poly(outer_meshes, merge_points=False, hole_size=1e6)
        outer_poly = (
            outer_poly
            .extract_surface()
            .triangulate()
            .clean(tolerance=1e-6)
            .fill_holes(1000)
            .clean(tolerance=1e-6)
        )
        core_poly = build_closed_poly(core_meshes, merge_points=False, hole_size=1e6)
        core_poly = (
            core_poly
            .extract_surface()
            .triangulate()
            .clean(tolerance=1e-6)
            .fill_holes(1000)
            .clean(tolerance=1e-6)
        )
        
        # Check whether surfaces are closed
        # outer_closed_bool = is_closed_surface(outer_poly)
        # core_closed_bool = is_closed_surface(core_poly)
        
        # Extract open edges for optional plotting
        # outer_open_edges = outer_poly.extract_feature_edges(boundary_edges=True)
        # core_open_edges = core_poly.extract_feature_edges(boundary_edges=True)
        
        # Slice intersection
        outer_slice_lines = outer_poly.slice(normal=(0, 1, 0), origin=(0, 0, 0))
        core_slice_lines = core_poly.slice(normal=(0, 1, 0), origin=(0, 0, 0))

        # Extract positive and negative slices
        
        outer_pts = outer_slice_lines.points
        outer_pos_pts = outer_pts[outer_pts[:, 2] >= 0]
        
        # Close cross-section by removing vertical points
        mask = (
            np.isclose(outer_pos_pts[:, 0], nacelle_profile.profile_segments[4][0, 0]) &
            (outer_pos_pts[:, 2] < nacelle_profile.profile_segments[4][0, 1])
        )
        outer_pos_pts = outer_pos_pts[~mask]
        # fig, ax = plt.subplots()
        # ax.scatter(outer_pos_pts[:, 0], outer_pos_pts[:, 2], marker='.')
        # plt.show()
        
        # outer_neg_pts = outer_pts[outer_pts[:, 2] <= 0]
        outer_neg_pts = outer_pos_pts.copy()
        outer_neg_pts[:, 2] *= -1
        
        outer_poly_pos = pv.PolyData(outer_pos_pts).delaunay_2d()
        outer_poly_neg = pv.PolyData(outer_neg_pts).delaunay_2d()
        
        core_pts = core_slice_lines.points
        core_poly = pv.PolyData(core_pts).delaunay_2d()
                
        if plot_mesh == True:
            plotter.show()
                
    # Extract camber of outer nacelle surface to be applied to radiator half-nacelle
    x_dim_camber = nacelle_profile.profile_segments[5][:, 0]
    y_dim_camber = nacelle_profile.profile_segments[5][:, 1]
    profile_x_min = np.min(np.hstack(profile_x))
    profile_x_max = np.max(np.hstack(profile_x))
    
    return (
        master_mesh, x_dim_camber, y_dim_camber, profile_x_min, profile_x_max, outer_poly_pos, outer_poly_neg, core_poly,
    )

#%%

if __name__ == '__main__':
    
    n_point_segment = 101
    r_hx_out = 20.5
    r_hx_in = 14
    r_hub_rotor = 6
    r_tip_rotor = 20
    l_rotor = 4
    l_stator = 2
    h_stator_in = 6
    h_stator_out = 8
    l_hx_side = 0
    # alpha_incl = np.deg2rad(35)
    alpha_incl = np.deg2rad(90)
    l_bp_nozzle = 18
    r_bp_nozzle_out = 19
    l_core_nozzle = 25
    r_core_nozzle_out = 9
    beta_core_nozzle = -np.deg2rad(10)
    beta_bp_nozzle = -np.deg2rad(10)
    l_spinner = 6
    l_intake = 20
    bypass_inner_angle = 0.0
    n_spanwise = 21
    save_dat = False
    plot_profile = True
    plot_mesh = False
    
    master_mesh, x_dim_camber, y_dim_camber, profile_x_min, profile_x_max, outer_poly_pos, outer_poly_neg, core_poly = generate_parallel_edf_geometry(
        n_point_segment,
        r_hx_out,
        r_hx_in,
        r_hub_rotor,
        r_tip_rotor,
        l_rotor,
        l_stator,
        h_stator_in,
        h_stator_out,
        l_hx_side,
        alpha_incl,
        l_bp_nozzle,
        r_bp_nozzle_out,
        l_core_nozzle,
        r_core_nozzle_out,
        beta_core_nozzle,
        beta_bp_nozzle,
        l_spinner,
        l_intake,
        bypass_inner_angle,
        n_spanwise,
        save_dat,
        plot_profile,
        plot_mesh,
    )
        
    #%%
    
    # Clip master mesh to negative y-values and extract slice
    master_mesh_clip = master_mesh.clip(
        normal=(0,1,0),
        origin=(0,0,0),
        invert=True,
    )
    master_mesh_slice = master_mesh.slice(
        normal=(0,1,0),
        origin=(0,0,0),
    )
    
    # Plot all features
    
    p = pv.Plotter()
    
    p.add_mesh(master_mesh_clip, show_edges=False, color='lightgrey', opacity=0.5)
    p.add_mesh(master_mesh_slice, color="black", line_width=3)
    
    # p.add_mesh(outer_poly, color="grey", opacity=0.6)
    # p.add_mesh(outer_open_edges, color="red", line_width=5)
    p.add_mesh(outer_poly_pos, color="red", opacity=1.0)
    p.add_mesh(outer_poly_neg, color="blue", opacity=1.0)
    
    # p.add_mesh(core_poly, color="grey", opacity=0.6)
    # p.add_mesh(core_open_edges, color="red", line_width=5)
    p.add_mesh(core_poly, color="green", opacity=1.0)
    
    p.add_axes()
    p.show_grid()
    p.show()
        
        
        
        
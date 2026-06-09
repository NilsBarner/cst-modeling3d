"""
This script executes test_edf_bypass_model_cst_backup.py ,which is superseded
by edf_bypass_model_cst.py and not to be edited, but kept for reference.
"""

__all__ = ["generate_bypass_edf_geometry"]

import os
import sys
import numpy as np
import pyvista as pv

from cst_modeling.io import output_surface
from cst_modeling.operation import Lofting_Revolution
from cst_modeling.tools.nils.bypass_edf_inclined_hx_profile import PoweredNacelleProfile  # inclined radiator
# from cst_modeling.tools.nils.bypass_edf_concentric_hx_profile import PoweredNacelleProfile  # concentric radiator
from nils.edf.shared_functions import build_closed_poly, is_closed_surface


def generate_bypass_edf_geometry(in_dict):
    
    circum_control_psi = [0.0, 90.0, 180.0, 270.0]
    
    # 2D profile
    profiles = []
    psi = circum_control_psi[0]
    nacelle_profile = PoweredNacelleProfile(psi=psi, n_point_segment=101)
    nacelle_profile.set_parameters(
        r_hx_out=in_dict['r_hx_out'],
        r_hx_in=in_dict['r_hx_in'],
        r_hub_rotor=in_dict['r_hub_rotor'],
        r_tip_rotor=in_dict['r_tip_rotor'],
        l_rotor=in_dict['l_rotor'],
        l_stator=in_dict['l_stator'],
        h_stator_in=in_dict['h_stator_in'],
        h_stator_out=in_dict['h_stator_out'],
        l_hx_side=in_dict['l_hx_side'],
        alpha_incl=in_dict['alpha_incl'],
        l_bp_nozzle=in_dict['l_bp_nozzle'],
        r_bp_nozzle_out=in_dict['r_bp_nozzle_out'],
        l_core_nozzle=in_dict['l_core_nozzle'],
        r_core_nozzle_out=in_dict['r_core_nozzle_out'],
        beta_core_nozzle=in_dict['beta_core_nozzle'],
        beta_bp_nozzle=in_dict['beta_bp_nozzle'],
        l_spinner=in_dict['l_spinner'],
        l_intake=in_dict['l_intake'],
        # r_in_spiral=in_dict['r_in_spiral'],
        # t_out_spiral=in_dict['t_out_spiral'],
        # N_spirals_t=in_dict['N_spirals_t'],
        cst_u=np.array([ 0.10, 0.10, 0.10, 0.10, 0.15]) * 2, # NILS: added factor 2
        cst_l=[-0.10, 0.15,-0.10, 0.05, 0.05],
        bypass_inner_angle=0.0,
        bypass_inner_control_points=[],
        core_outer_control_points=[],
        core_inner_control_points=[],
    )
    profile_x, profile_y = nacelle_profile.get_profile()
    # profiles.append([np.concatenate(profile_x), np.concatenate(profile_y)])  # cannot concatenate into one single array, as sub-arrays not ordered, resulting in weird connecting lines (see below plots)
    profiles.append([profile_x, profile_y])
    
    # Plot 2D profile(s)
    
    if in_dict['plot_profile'] == True:
        nacelle_profile.plot(show=True)
        # nacelle_profile.plot_side_view()
        # fig, ax = plt.subplots()
        # ax.plot(np.concatenate(profile_x), np.concatenate(profile_y))
        # plt.show()
        # # vs
        # fig, ax = plt.subplots()
        # for i in range(len(profile_x)):
        #     ax.plot(profile_x[i], profile_y[i])
        # plt.show()
    
    # 3D axisymmetric body of revolution
    
    path = os.path.dirname(sys.argv[0])
    
    surfs_list = []
    section_s_loc = [0.0, 0.25, 0.50, 0.75]
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
                n_spanwise=21,
            )
            
            # # Plot individual sub-profiles
            # fig, ax = plt.subplots()
            # ax.plot(sub_profile_x, sub_profile_y)
            # plt.show()
            
            surfs = nacelle.sweep(interp_profile_kind='linear')
            surfs_list.append(surfs)
    
        if in_dict['plot_mesh'] == True:
            pv.global_theme.allow_empty_mesh = True
            plotter = pv.Plotter()
    
        all_meshes = []
    
        # Create .dat file from surface generated from each revolved sub-profile
        for i, surfs in enumerate(surfs_list):
            for i_surf, surf in enumerate(surfs):
                
                if in_dict['save_dat'] == True:
                    output_surface(surf, fname=os.path.join(path, f'surface_{i}.dat'), ID=i_surf)
                    
                mesh = pv.StructuredGrid(*surf)
                    
                # store mesh
                all_meshes.append(mesh)
                    
                if in_dict['plot_mesh'] == True:
                    mesh = pv.StructuredGrid(*surf)  # create VTK dataset from surface coordinates
                    plotter.add_mesh(mesh, show_edges=True)
                    
        if in_dict['plot_mesh'] == True:
            plotter.show()
    
        master_mesh = pv.merge(all_meshes)
        
        # Create separate surfaces for nacelle cross-sections and fan
        
        # fig, ax = plt.subplots()
        # ax.plot(nacelle_profile.profile_segments[50][:, 0], nacelle_profile.profile_segments[50][:, 1])
        # ax.plot(nacelle_profile.profile_segments[26][:, 0], nacelle_profile.profile_segments[26][:, 1])
        # ax.plot(nacelle_profile.profile_segments[12][:, 0], nacelle_profile.profile_segments[12][:, 1])
        # ax.plot(nacelle_profile.profile_segments[14][:, 0], nacelle_profile.profile_segments[14][:, 1])
        # plt.show()
        
        # Define solid-body surfaces
        outer_surfs_list = [
            surfs_list[nacelle_profile.profile_mapping[8]],
            surfs_list[nacelle_profile.profile_mapping[7]],
            surfs_list[nacelle_profile.profile_mapping[10]],
        ]
        inner_surfs_list = [
            surfs_list[nacelle_profile.profile_mapping[4]],
            surfs_list[nacelle_profile.profile_mapping[5]],
            surfs_list[nacelle_profile.profile_mapping[20]],
        ]
        core_surfs_list = [
            surfs_list[nacelle_profile.profile_mapping[0]],
            surfs_list[nacelle_profile.profile_mapping[1]],
            surfs_list[nacelle_profile.profile_mapping[2]],
            surfs_list[nacelle_profile.profile_mapping[29]],
            surfs_list[nacelle_profile.profile_mapping[22]],
        ]
        hx_surfs_list = [
            surfs_list[nacelle_profile.profile_mapping[50]],
            surfs_list[nacelle_profile.profile_mapping[26]],
            surfs_list[nacelle_profile.profile_mapping[12]],
            surfs_list[nacelle_profile.profile_mapping[14]],
        ]
        
        # Convert surfaces to meshes
        outer_meshes = list(np.concatenate([
            [pv.StructuredGrid(*surf) for surf in outer_surfs] for outer_surfs in outer_surfs_list
        ]))
        inner_meshes = list(np.concatenate([
            [pv.StructuredGrid(*surf) for surf in inner_surfs] for inner_surfs in inner_surfs_list
        ]))
        core_meshes = list(np.concatenate([
            [pv.StructuredGrid(*surf) for surf in core_surfs] for core_surfs in core_surfs_list
        ]))
        hx_meshes = list(np.concatenate([
            [pv.StructuredGrid(*surf) for surf in hx_surfs] for hx_surfs in hx_surfs_list
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
        inner_poly = build_closed_poly(inner_meshes, merge_points=False, hole_size=1e6)
        inner_poly = (
            inner_poly
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
        hx_poly = build_closed_poly(hx_meshes, merge_points=False, hole_size=1e6)
        hx_poly = (
            hx_poly
            .extract_surface()
            .triangulate()
            .clean(tolerance=1e-6)
            .fill_holes(1000)
            .clean(tolerance=1e-6)
        )
        
        # Check whether surfaces are closed
        # outer_closed_bool = is_closed_surface(outer_poly)
        # inner_closed_bool = is_closed_surface(inner_poly)
        # core_closed_bool = is_closed_surface(core_poly)
        
        # Extract open edges for optional plotting
        # outer_open_edges = outer_poly.extract_feature_edges(boundary_edges=True)
        # inner_open_edges = inner_poly.extract_feature_edges(boundary_edges=True)
        # core_open_edges = core_poly.extract_feature_edges(boundary_edges=True)
        
        # Slice intersection
        outer_slice_lines = outer_poly.slice(normal=(0, 1, 0), origin=(0, 0, 0))
        inner_slice_lines = inner_poly.slice(normal=(0, 1, 0), origin=(0, 0, 0))
        core_slice_lines = core_poly.slice(normal=(0, 1, 0), origin=(0, 0, 0))
        hx_slice_lines = hx_poly.slice(normal=(0, 1, 0), origin=(0, 0, 0))

        # Extract positive and negative slices
        
        outer_pts = outer_slice_lines.points
        outer_pos_pts = outer_pts[outer_pts[:, 2] >= 0]
        outer_neg_pts = outer_pts[outer_pts[:, 2] <= 0]
        outer_poly_pos = pv.PolyData(outer_pos_pts).delaunay_2d()
        outer_poly_neg = pv.PolyData(outer_neg_pts).delaunay_2d()
        
        inner_pts = inner_slice_lines.points
        inner_pos_pts = inner_pts[inner_pts[:, 2] >= 0]
        inner_neg_pts = inner_pts[inner_pts[:, 2] <= 0]
        inner_poly_pos = pv.PolyData(inner_pos_pts).delaunay_2d()
        inner_poly_neg = pv.PolyData(inner_neg_pts).delaunay_2d()
        
        core_pts = core_slice_lines.points
        core_poly = pv.PolyData(core_pts).delaunay_2d()
        
        hx_pts = hx_slice_lines.points
        hx_pos_pts = hx_pts[hx_pts[:, 2] >= 0]
        hx_neg_pts = hx_pts[hx_pts[:, 2] <= 0]
        hx_poly_pos = pv.PolyData(hx_pos_pts).delaunay_2d()
        hx_poly_neg = pv.PolyData(hx_neg_pts).delaunay_2d()
        
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
        hx_poly_clip = hx_poly.clip(
            normal=(0,1,0),
            origin=(0,0,0),
            invert=True,
        )
        
        # Plot all features
        
        p = pv.Plotter()
        
        p.add_mesh(master_mesh_clip, show_edges=False, color='lightgrey', opacity=0.5)
        p.add_mesh(master_mesh_slice, color="black", line_width=3)
        
        # p.add_mesh(outer_poly, color="grey", opacity=0.6)
        # p.add_mesh(outer_open_edges, color="red", line_width=5)
        p.add_mesh(outer_poly_pos, color="grey", opacity=1.0)
        p.add_mesh(outer_poly_neg, color="grey", opacity=1.0)
        
        # p.add_mesh(inner_poly, color="grey", opacity=0.6)
        # p.add_mesh(inner_open_edges, color="red", line_width=5)
        p.add_mesh(inner_poly_pos, color="grey", opacity=1.0)
        p.add_mesh(inner_poly_neg, color="grey", opacity=1.0)
        
        p.add_mesh(hx_poly_pos, color="orange", opacity=1.0)
        p.add_mesh(hx_poly_neg, color="orange", opacity=1.0)
        
        # p.add_mesh(core_poly, color="grey", opacity=0.6)
        # p.add_mesh(core_open_edges, color="red", line_width=5)
        p.add_mesh(core_poly, color="grey", opacity=1.0)
        
        p.add_mesh(hx_poly_clip, show_edges=False, color='orange', opacity=0.5)
        
        p.add_axes()
        p.show_grid()
        p.show()
        
        out_dict = {
            "master": master_mesh,
            "outer_pos": outer_poly_pos,
            "outer_neg": outer_poly_neg,
            "inner_pos": inner_poly_pos,
            "inner_neg": inner_poly_neg,
            "hx_pos": hx_poly_pos,
            "hx_neg": hx_poly_neg,
            "hx": hx_poly,
            "core_poly": core_poly,
        }
        
        return out_dict


"""
This script parametrises and plots an isolated axial ducted radiator. It uses
cst_modeling.tools.axial_ducted_radiator_nils, which has been adapted from
the cst_modeling\tools\nacelle.py script, to parametrise the 2D horizontal
and vertical profiles, and guide_curve_getter_final.py, which is inspired by
low_paper_2008_univparamgeomreprmeth_kulfan.py to parametrise the cross-section.
"""

__all__ = ["generate_axial_ducted_radiator_geometry"]

import os
import sys
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from cst_modeling.io import output_surface
from cst_modeling.basic import BasicSection, BasicSurface
from cst_modeling.tools.nils.axial_ducted_radiator_profile import PoweredNacelleProfile
from nils.eprop.axial.guide_curve_getter import get_guide_curve
from nils.eprop.shared_functions import Section, curve, order_curve3d
from nils.edf.shared_functions import build_closed_poly, is_closed_surface


def generate_axial_ducted_radiator_geometry(
    l,
    w_intake, h_intake,
    l_up_duct,
    l_hx, w_hx, h_hx,
    l_down_duct,
    l_fan, d_fan,
    l_nozzle, d_nozzle,
    N_crosssect=100,
    plot_profile=False,
    plot_scatter=False,
    plot_mesh=False,
    save_dat=False,
    dx=0.0,
    dy=0.0,
    dz=0.0,
):
    
    # Upper- and lower-half reference radii and angles (intake, hx, nozzle, camber bump)
    _, _, _, r_u_intake, r_l_intake, theta_u_intake, theta_l_intake = get_guide_curve(
        0.1, 0.1, w_intake, h_intake, N=N_crosssect
    )
    _, _, _, r_u_hx, r_l_hx, theta_u_hx, theta_l_hx = get_guide_curve(
        0.1, 0.1, w_hx, h_hx, N=N_crosssect
    )
    _, _, _, r_u_camber_bump, r_l_camber_bump, theta_u_camber_bump, theta_l_camber_bump = get_guide_curve(
        # 0.1, 0.1, 0.15, 0.06, N=N_crosssect
        0.1, 0.1, 0.15 * 1.5, 0.06 * 1.5, N=N_crosssect
    )
    _, _, _, r_u_nozzle, r_l_nozzle, theta_u_nozzle, theta_l_nozzle = get_guide_curve(
        0.1, 0.1, d_nozzle/2, d_nozzle/2, N=N_crosssect
    )

    # Stack upper- and lower-half radii
    r_intake_curve = np.hstack((r_u_intake, r_l_intake[:-1][::-1]))  # close cross-section, but do no duplicate intermediate point
    r_hx_curve = np.hstack((r_u_hx, r_l_hx[:-1][::-1]))
    r_camber_bump_curve = np.hstack((r_u_camber_bump, r_l_camber_bump[:-1][::-1]))
    r_nozzle_curve = np.hstack((r_u_nozzle, r_l_nozzle[:-1][::-1]))
    
    # Stack upper- and lower-half angles
    theta_intake_curve = np.hstack((theta_u_intake, theta_l_intake[:-1][::-1]))  # close cross-section, but do no duplicate intermediate point
    theta_hx_curve = np.hstack((theta_u_hx, theta_l_hx[:-1][::-1]))
    theta_camber_bump_curve = np.hstack((theta_u_camber_bump, theta_l_camber_bump[:-1][::-1]))
    theta_nozzle_curve = np.hstack((theta_u_nozzle, theta_l_nozzle[:-1][::-1]))
    
    if plot_scatter:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')  
    
    profiles = []
    section_theta_list = []  # record the per-section theta used for each saved profile
    
    for i_sec in range(N_crosssect):
        
        # Extract reference radius of current profile
        r_intake = r_intake_curve[i_sec]
        r_hx = r_hx_curve[i_sec]
        r_camber_bump = r_camber_bump_curve[i_sec]
        r_nozzle = r_nozzle_curve[i_sec]
        
        # Extract reference angle of current profile
        theta_intake = theta_intake_curve[i_sec]
        theta_hx = theta_hx_curve[i_sec]
        theta_camber_bump = theta_camber_bump_curve[i_sec]
        theta_nozzle = theta_nozzle_curve[i_sec]
    
        # Create 2D profile
        nacelle_profile = PoweredNacelleProfile(psi=None, n_point_segment=101)
        nacelle_profile.set_parameters(
            r_spinner=0.0, theta_spinner=90.0, r_fan=r_hx/2,
            highlight_x=-l_up_duct, highlight_y=r_intake/2,
            intake_face_center=(-l_up_duct, 0),
            l_nacelle=l, r_te=d_nozzle/4,  # circular nozzle
            # l_nacelle=l, r_te=r_nozzle,  # rectangular nozzle
            l_fan=l_hx, r_bypass_outer=r_hx/2, r_bypass_inner=0.0,
            x_core_cowl_0=None, y_core_cowl_0=None,
            x_core_cowl_1=None, y_core_cowl_1=None,
            x_core_duct=None, r_core_outer=None, r_core_inner=None,
            x_core_plug_0=None, y_core_plug_0=None, x_core_plug_1=None,
            cst_u=[ 0.10, 0.10, 0.10, 0.10, 0.10], 
            cst_l=[-0.10, 0.15,-0.10, 0.05, 0.05],
            bypass_inner_angle=0.0,
            bypass_inner_control_points=[],
            core_outer_control_points=[],
            core_inner_control_points=[],
            h_camber_bump = r_camber_bump,
            xc_camber_bump = 0.4,
        )
        
        # Log 2D profile coordinates and angle with axis of revolution
        profile_x, profile_y = nacelle_profile.get_profile()
        profiles.append([profile_x, profile_y])
        section_theta_list.append(theta_intake)
        
        if plot_scatter:
            ax.scatter(profile_x, profile_y * np.cos(np.deg2rad(theta_intake)), profile_y * np.sin(np.deg2rad(theta_intake)), color='k', marker='.')
            # ax.scatter(profile_x, profile_y * np.cos(np.deg2rad(theta_intake)), -profile_y * np.sin(np.deg2rad(theta_intake)), color='k', marker='.')
        
    if plot_profile == True:
        nacelle_profile.plot(show=True)
        
    assert len(profiles) == len(section_theta_list), "profiles and section_theta_list must match length"
    # print('section_theta_list, np.diff(section_theta_list) =', section_theta_list, np.diff(section_theta_list))
    
    if plot_scatter:
        ax.set_aspect('equal')
        plt.show()
    
    # Create closed 3D surface by lofting 2D profiles around perpendicular guide curve
    
    # --- Minimal addition: append the mirrored/reversed half so full-body is constructed ---
    # At this point: `profiles` has N entries and `section_theta_list` has N entries.
    # We append a reversed copy of the profiles and the negated reversed angles (mod 360).
    orig_profiles = profiles[:]                     # length N
    
    # --- Minimal replacement: create uniform full-circle angles and mirror profiles ---
    N_orig = len(orig_profiles)
    # make full-circle angles uniformly spaced (2*N points)
    theta_full = np.linspace(0.0, 360.0, 2 * N_orig, endpoint=False)
    
    # make mirrored profiles so circumferential order is continuous:
    # original (0..~180) then reversed original (to sweep back 180..360)
    profiles_full = orig_profiles + orig_profiles[::-1]
    
    # replace lists with the full versions
    profiles = profiles_full
    section_theta_list = list(theta_full)
    
    # Now continue with building the closed surface as before...
    n_sections = len(profiles)
    n_sub = len(profiles[0][0])          # number of sub-profiles
    n_point = profiles[0][0][0].shape[0]
    n_span = n_sections + 1
    
    all_meshes = []
    surfs_list = []
    for j in range(n_sub):   # loop over sub-profiles
    
        surf_x = np.zeros((n_span, n_point))
        surf_y = np.zeros((n_span, n_point))
        surf_z = np.zeros((n_span, n_point))
    
        for i_local in range(n_span):
    
            idx = i_local % n_sections
            profile_x_list, profile_y_list = profiles[idx]
    
            sub_profile_x = profile_x_list[j]
            sub_profile_y = profile_y_list[j]
    
            theta_deg = section_theta_list[idx]
            theta_rad = np.deg2rad(theta_deg)
    
            surf_x[i_local, :] = sub_profile_x + dx
            surf_y[i_local, :] = sub_profile_y * np.cos(theta_rad) + dy
            surf_z[i_local, :] = sub_profile_y * np.sin(theta_rad) + dz
    
        surfs = [surf_x, surf_y, surf_z]
        surfs_list.append(surfs)
    
        mesh = pv.StructuredGrid(surf_x, surf_y, surf_z)
        all_meshes.append(mesh)
    
    master_mesh = pv.merge(all_meshes)
    
    # =============================================================================
    # =============================================================================
    surface = BasicSurface(n_sec=2, name='out', nn=201, ns=51, projection=False)
    
    x_nozzle = np.max(master_mesh.points[:, 0])
    # NILS: random dimensions!
    d_fan = d_nozzle/2
    l_fan = l_nozzle/4
    
    ### NILS: fan inlet
    
    i = 0
    
    x_centroid_fan_inlet = x_nozzle - l_fan
    y_centroid_fan_inlet = 0.0
    z_centroid_fan_inlet = dz
    
    surface.secs[i] = Section(n_curve = 4)
    surface.secs[i].circle(
        np.array([y_centroid_fan_inlet - d_fan/2, z_centroid_fan_inlet]),
        np.array([y_centroid_fan_inlet + d_fan/2, z_centroid_fan_inlet]),
        np.array([y_centroid_fan_inlet, z_centroid_fan_inlet + d_fan/2]),
        nn=500,
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
    
    x_centroid_fan_outlet = x_centroid_fan_inlet + l_fan
    y_centroid_fan_outlet = y_centroid_fan_inlet
    z_centroid_fan_outlet = z_centroid_fan_inlet
    
    surface.secs[i] = Section(n_curve = 4)
    surface.secs[i].circle(
        np.array([y_centroid_fan_outlet - d_fan/2, z_centroid_fan_outlet]),
        np.array([y_centroid_fan_outlet + d_fan/2, z_centroid_fan_outlet]),
        np.array([y_centroid_fan_outlet, z_centroid_fan_outlet + d_fan/2]),
        nn=500,
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
    
    surface.geo(update_sec=False)
    surface.flip(axis = '+Y')
    surface.flip(axis = '+X')
    
    fan_mesh = pv.merge([pv.StructuredGrid(*surf) for surf in surface.surfs])
    # =============================================================================
    # =============================================================================
    
    # =============================================================================
    # Define solid-body surfaces
    outer_surfs_list = [
        surfs_list[nacelle_profile.profile_mapping[3]],
        surfs_list[nacelle_profile.profile_mapping[2]],
        surfs_list[nacelle_profile.profile_mapping[4]],
        surfs_list[nacelle_profile.profile_mapping[7]],
    ]
    
    hx_surfs_list = [
        surfs_list[nacelle_profile.profile_mapping[0]],
        surfs_list[nacelle_profile.profile_mapping[1]],
        surfs_list[nacelle_profile.profile_mapping[5]],
        surfs_list[nacelle_profile.profile_mapping[6]],
        surfs_list[nacelle_profile.profile_mapping[7]],
    ]
    
    # outer_meshes = list(np.concatenate([
    #     [pv.StructuredGrid(*surf) for surf in outer_surfs] for outer_surfs in outer_surfs_list
    # ]))
    outer_meshes = [pv.StructuredGrid(*surf) for surf in outer_surfs_list]
    # hx_meshes = list(np.concatenate([
    #     [pv.StructuredGrid(*surf) for surf in hx_surfs] for hx_surfs in hx_surfs_list
    # ]))
    hx_meshes = [pv.StructuredGrid(*surf) for surf in hx_surfs_list]
    
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
    
    # Extract open edges for optional plotting
    # outer_open_edges = outer_poly.extract_feature_edges(boundary_edges=True)
    
    # Slice intersection
    outer_slice_lines = outer_poly.slice(normal=(0, 1, 0), origin=(0, 0, 0))
    hx_slice_lines = hx_poly.slice(normal=(0, 1, 0), origin=(0, 0, 0))

    # Extract positive and negative slices
    
    outer_pts = outer_slice_lines.points
    outer_pos_pts = outer_pts[outer_pts[:, 2] >= 0]
    
    # =============================================================================
    mask = (
        np.isclose(outer_pos_pts[:, 0], nacelle_profile.profile_segments[4][0, 0]) &
        (outer_pos_pts[:, 2] < nacelle_profile.profile_segments[4][0, 1])
    )
    outer_pos_pts = outer_pos_pts[~mask]
    
    # fig, ax = plt.subplots()
    # ax.scatter(outer_pos_pts[:, 0], outer_pos_pts[:, 2], marker='.')
    # plt.show()
    # sys.exit()
    # =============================================================================
    
    # outer_neg_pts = outer_pts[outer_pts[:, 2] <= 0]
    # =============================================================================
    outer_neg_pts = outer_pos_pts.copy()
    outer_neg_pts[:, 2] *= -1
    # =============================================================================
    
    outer_poly_pos = pv.PolyData(outer_pos_pts).delaunay_2d()
    outer_poly_neg = pv.PolyData(outer_neg_pts).delaunay_2d()
    
    hx_pts = hx_slice_lines.points
    hx_pos_pts = hx_pts[hx_pts[:, 2] >= 0]
    hx_neg_pts = hx_pts[hx_pts[:, 2] <= 0]
    hx_poly_pos = pv.PolyData(hx_pos_pts).delaunay_2d()
    hx_poly_neg = pv.PolyData(hx_neg_pts).delaunay_2d()
    # =============================================================================
    
    if plot_mesh == True:
    
        # mesh = pv.read('axial_ducted_radiator_mesh.dat')
        
        # master_mesh.plot(border=True, border_color='k', show_axes=True, show_bounds=True)
        
        plotter = pv.Plotter()
        plotter.add_mesh(all_meshes[0], color='red', opacity=0.2)
        plotter.add_mesh(all_meshes[1], color='blue', opacity=0.2)
        plotter.add_mesh(all_meshes[2], color='green', opacity=0.2)
        plotter.add_mesh(all_meshes[3], color='orange', opacity=0.2)
        plotter.add_mesh(all_meshes[4], color='brown', opacity=0.2)
        plotter.add_mesh(all_meshes[5], color='purple', opacity=0.2)
        plotter.add_mesh(all_meshes[6], color='grey', opacity=0.2)
        plotter.add_mesh(all_meshes[7], color='yellow', opacity=0.2)
        
        plotter.add_mesh(fan_mesh, color='black', opacity=1.0)
        
        plotter.add_axes()
        plotter.show_grid()
        plotter.show()
    
    return master_mesh, hx_poly_pos, hx_poly_neg, hx_poly, fan_mesh

#%%

if __name__ == '__main__':
    
    l = 3.2
    w = 1.2
    h = 0.6
    w_intake = 0.6
    h_intake = 0.3
    l_up_duct = 1.5
    l_hx = 0.3
    w_hx = 1.2
    h_hx = 0.6
    l_down_duct = 0.8
    l_fan = 0.3
    d_fan = 0.5
    l_nozzle = 0.3
    d_nozzle = 0.3
    
    # l = 40
    # w_intake = 4 * 4
    # h_intake = 4 * 2
    # l_up_duct = 15
    # l_hx = 5
    # w_hx = 6 * 4
    # h_hx = 6 * 2
    # l_down_duct = 10
    # l_fan = 5
    # d_fan = 5 * 2  # NILS: not used
    # l_nozzle = 5
    # d_nozzle = 4 * 2
    
    generate_axial_ducted_radiator_geometry(
        l,
        w_intake, h_intake,
        l_up_duct,
        l_hx, w_hx, h_hx,
        l_down_duct,
        l_fan, d_fan,
        l_nozzle, d_nozzle,
        plot_mesh=True,
    )

'''
Non-axisymmetric Powered Engine Nacelle (PEN)
'''
import os
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt

from cst_modeling.io import output_surface
from cst_modeling.operation import Lofting_Revolution
# from cst_modeling.tools.nacelle import NacelleIntakeHighlight, PoweredNacelleProfile
from cst_modeling.tools.axial_ducted_radiator_nils import NacelleIntakeHighlight, PoweredNacelleProfile


def parametrise_axial_ducted_radiator(
    l, w, h,
    w_intake, h_intake,
    l_up_duct,
    l_hx, w_hx, h_hx,
    l_down_duct,
    l_fan, d_fan,
    l_nozzle, d_nozzle,
):

    path = os.path.dirname(sys.argv[0])
    
    circum_control_psi = [0.0, 90.0, 180.0, 270.0]
    
    #* Nacelle Profile
    
    profiles = []
    
    r_scale_array = np.array([
        [w/h],
        [1],
        [w/h],
        [1],
    ])
    
    for i_sec in range(len(circum_control_psi)):
        
        psi = circum_control_psi[i_sec]
    
        nacelle_profile = PoweredNacelleProfile(psi=psi, n_point_segment=101)
        
        nacelle_profile.set_parameters(
            # r_spinner=0.0, theta_spinner=90.0, r_fan=6,
            # highlight_x=-15, highlight_y=4,
            # intake_face_center=(-15, 0),
            # l_nacelle=40, r_te=4,
            # l_fan=5.0, r_bypass_outer=6, r_bypass_inner=0.0,
            # x_core_cowl_0=None, y_core_cowl_0=None,
            # x_core_cowl_1=None, y_core_cowl_1=None,
            # x_core_duct=None, r_core_outer=None, r_core_inner=None,
            # x_core_plug_0=None, y_core_plug_0=None, x_core_plug_1=None,
            # cst_u=[ 0.10, 0.10, 0.10, 0.10, 0.10], 
            # cst_l=[-0.10, 0.15,-0.10, 0.05, 0.05],
            # bypass_inner_angle=0.0,
            # bypass_inner_control_points=[],
            # core_outer_control_points=[],
            # core_inner_control_points=[],
            r_spinner=0.0, theta_spinner=90.0, r_fan=h_hx/2,
            highlight_x=-l_up_duct, highlight_y=h_intake/2,
            intake_face_center=(-l_up_duct, 0),
            l_nacelle=l, r_te=d_nozzle/2,
            l_fan=l_hx, r_bypass_outer=h_hx/2, r_bypass_inner=0.0,
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
        )
        
        # =============================================================================
        #         profile_x, profile_y = nacelle_profile.get_profile()
        #         
        #         profiles.append([profile_x, profile_y])
        
        
        # profile_x, profile_y = nacelle_profile.get_profile()

        # # per-section radial scaling (vertical / radial direction)
        # # r_scale_array must be length n_sections (or shape [n_sections,1])
        # y_scale = float(r_scale_array[i_sec][0])  # or r_scale_array[i_sec,0]
        # print('y_scale =', y_scale)
        
        # profile_y_scaled = profile_y * y_scale
        
        # profiles.append([profile_x, profile_y_scaled])
        
        
        # profile_x, profile_y = nacelle_profile.get_profile()
        
        # # after profile_x, profile_y = nacelle_profile.get_profile()
        # # enforce vertical half-height (h/2)
        # desired_half_height = h / 2.0
        
        # max_py = np.max(np.abs(profile_y))
        # if max_py > 0:
        #     profile_y_scaled = profile_y * (desired_half_height / max_py)
        # else:
        #     profile_y_scaled = profile_y.copy()
        
        # profiles.append([profile_x, profile_y_scaled])
        
        
        # =============================================================================
        # =============================================================================
        # --- replace the current per-profile scaling block with this ---
        profile_x, profile_y = nacelle_profile.get_profile()
        
        # optional: make profile touch the axis (so the rotated surface can close)
        profile_y = profile_y - np.min(profile_y)
        
        # normalize radial amplitude to 1.0 (so loft will scale to absolute sizes)
        max_py = np.max(np.abs(profile_y))
        if max_py > 0.0:
            profile_y_norm = profile_y / max_py
        else:
            profile_y_norm = profile_y.copy()
        
        profiles.append([profile_x, profile_y_norm])
        # --- end replacement ---
        # =============================================================================
        # =============================================================================
        # =============================================================================
        
        if i_sec == len(circum_control_psi) - 1:
            nacelle_profile.plot(show=True)
    
    
    #* Nacelle surface of revolution
    
    # nacelle = Lofting_Revolution(
    #     profiles=profiles,
    #     section_s_loc=[a/360.0 for a in circum_control_psi],
    #     section_x=0.0,
    #     section_radius=0.0,
    #     section_scale=1.0,
    #     n_spanwise=51,
    # )
    
    # =============================================================================
    section_s_loc=[0.0, 0.25, 0.50, 0.75]
    
    # =============================================================================
    max_profile_y = max(np.max(np.abs(p[1])) for p in profiles)
    # =============================================================================
    # nacelle = Lofting_Revolution(
    #     profiles=profiles,
    #     section_s_loc=section_s_loc,  # [a/360.0 for a in circum_control_psi]
    #     section_x=0.0,
    #     # section_radius=0.25 * max_profile_y,  # 100,  # 0.0
    #     section_radius=w / 2.0,
    #     section_scale=1.0,
    #     n_spanwise=21,  # 51,
    #     section_shape='superellipse',   # <--- new
    #     superellipse_exp=5.0,           # <--- exponent = 5
    # )
    
    
    # # nacelle = Lofting_Revolution(
    # #     profiles=profiles,
    # #     section_s_loc=section_s_loc,
    # #     section_x=0.0,
    # #     section_radius = w / 2.0,      # <-- use half width here
    # #     section_scale=1.0,
    # #     n_spanwise=181,               # <-- many samples to remove lobes
    # #     section_shape='superellipse',
    # #     superellipse_exp=5.0,
    # # )
    
    
    nacelle = Lofting_Revolution(
        profiles=profiles,
        section_s_loc=section_s_loc,
        section_x=0.0,
        section_radius = w / 2.0,       # a = half width (keeps backward-compatible single value)
        section_radius_y = h / 2.0,     # NEW: half height (b)
        section_scale=1.0,
        n_spanwise=181,                 # use many samples to avoid lobes
        section_shape='superellipse',
        superellipse_exp=5.0,
    )
    
    # sweeping: use periodic interpolation
    surfs = nacelle.sweep(interp_profile_kind='periodic')
    # =============================================================================

    # =============================================================================
    # surfs = nacelle.sweep(interp_profile_kind='periodic')
    surfs = nacelle.sweep(interp_profile_kind='linear')
    # =============================================================================

    for i_surf, surf in enumerate(surfs):
        output_surface(surf, fname=os.path.join(path, 'nacelle-non-axisymmetric.dat'), ID=i_surf)
    
    # NILS
    
    import pyvista as pv
    
    mesh = pv.read(os.path.join(path, 'nacelle-non-axisymmetric.dat'))
    cpos = mesh.plot(border=True, border_color='k', show_axes=True)
    
    return

#%%

if __name__ == '__main__':
    
    # l = 3.2
    # w = 1.2
    # h = 0.6
    # w_intake = 0.6
    # h_intake = 0.3
    # l_up_duct = 1.5
    # l_hx = 0.3
    # w_hx = 1.2
    # h_hx = 0.6
    # l_down_duct = 0.8
    # l_fan = 0.3
    # d_fan = 0.5
    # l_nozzle = 0.3
    # d_nozzle = 0.3
    
    l = 40
    w = 7.5 * 4
    h = 7.5 * 2
    w_intake = 4 * 2
    h_intake = 4 * 2 
    l_up_duct = 15
    l_hx = 5
    w_hx = 6 * 2
    h_hx = 6 * 2
    l_down_duct = 10
    l_fan = 5
    d_fan = 5 * 2  # NILS: not used
    l_nozzle = 5
    d_nozzle = 4 * 2
    
    parametrise_axial_ducted_radiator(
        l, w, h,
        w_intake, h_intake,
        l_up_duct,
        l_hx, w_hx, h_hx,
        l_down_duct,
        l_fan, d_fan,
        l_nozzle, d_nozzle,
    )
    
    
    
    


'''
Non-axisymmetric Powered Engine Nacelle (PEN)
'''
import os
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt

from cst_modeling.io import output_surface
from cst_modeling.operation_org import Lofting_Revolution
# from cst_modeling.tools.nacelle import NacelleIntakeHighlight, PoweredNacelleProfile
from cst_modeling.tools.axial_ducted_radiator_nils import NacelleIntakeHighlight, PoweredNacelleProfile


def parametrise_axial_ducted_radiator(
    l,# w, h,
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
    
    # =============================================================================
    r_intake = [w_intake, h_intake, w_intake, h_intake]
    r_hx = [w_hx, h_hx, w_hx, h_hx]
    h_camber_bump_list = [0.15, 0.06, 0.15, 0.06]
    # =============================================================================
    
    profiles = []
    
    for i_sec in range(len(circum_control_psi)):
        
        psi = circum_control_psi[i_sec]
    
        nacelle_profile = PoweredNacelleProfile(psi=psi, n_point_segment=101)
        
        # nacelle_profile.set_parameters(
        #     r_spinner=0.0, theta_spinner=90.0, r_fan=h_hx/2,
        #     highlight_x=-l_up_duct, highlight_y=h_intake/2,
        #     intake_face_center=(-l_up_duct, 0),
        #     l_nacelle=l, r_te=d_nozzle/2,
        #     l_fan=l_hx, r_bypass_outer=h_hx/2, r_bypass_inner=0.0,
        #     x_core_cowl_0=None, y_core_cowl_0=None,
        #     x_core_cowl_1=None, y_core_cowl_1=None,
        #     x_core_duct=None, r_core_outer=None, r_core_inner=None,
        #     x_core_plug_0=None, y_core_plug_0=None, x_core_plug_1=None,
        #     cst_u=[ 0.10, 0.10, 0.10, 0.10, 0.10], 
        #     cst_l=[-0.10, 0.15,-0.10, 0.05, 0.05],
        #     bypass_inner_angle=0.0,
        #     bypass_inner_control_points=[],
        #     core_outer_control_points=[],
        #     core_inner_control_points=[],
        # )
        
        # =============================================================================
        nacelle_profile.set_parameters(
            r_spinner=0.0, theta_spinner=90.0, r_fan=r_hx[i_sec]/2,
            highlight_x=-l_up_duct, highlight_y=r_intake[i_sec]/2,
            intake_face_center=(-l_up_duct, 0),
            l_nacelle=l, r_te=d_nozzle/2,
            l_fan=l_hx, r_bypass_outer=r_hx[i_sec]/2, r_bypass_inner=0.0,
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
            # =============================================================================
            h_camber_bump = h_camber_bump_list[i_sec],
            xc_camber_bump = 0.4,
            # =============================================================================
        )
        # =============================================================================
        
        profile_x, profile_y = nacelle_profile.get_profile()
        
        profiles.append([profile_x, profile_y])
        
        # if i_sec == len(circum_control_psi) - 1:
        nacelle_profile.plot(show=True)
        
        print('np.max(profile_y) =', np.max(profile_y))
        
    # sys.exit()
        
    #* Nacelle surface of revolution
    
    # nacelle = Lofting_Revolution(
    #     profiles=profiles,
    #     section_s_loc=[a/360.0 for a in circum_control_psi],
    #     section_x=0.0,
    #     section_radius=0.0,
    #     section_scale=1.0,
    #     n_spanwise=51,
    # )
    
    nacelle = Lofting_Revolution(
        profiles=profiles,
        section_s_loc=[a/360.0 for a in circum_control_psi],
        section_x=0.0,
        section_radius=0.0,          # kept for backwards compatibility; ignored when width/height used
        section_scale=1.0,
        # --- NEW: build around a superellipse of exponent 5, max width w and max height h ---
        # section_width = w,
        # section_height = h,
        # =============================================================================
        section_width = None,
        section_height = None,
        # =============================================================================
        superellipse_exponent = 5.0,
        n_spanwise=51,
    )

    # surfs = nacelle.sweep(interp_profile_kind='periodic')
    surfs = nacelle.sweep(interp_profile_kind='linear')

    for i_surf, surf in enumerate(surfs):
        output_surface(surf, fname=os.path.join(path, 'nacelle-non-axisymmetric.dat'), ID=i_surf)
    
    # NILS
    
    import pyvista as pv
    
    mesh = pv.read(os.path.join(path, 'nacelle-non-axisymmetric.dat'))
    cpos = mesh.plot(border=True, border_color='k', show_axes=True, show_bounds=True)
    
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
    # w = 7.5 * 4
    # h = 7.5 * 2
    w_intake = 4 * 4  # 2
    h_intake = 4 * 2
    l_up_duct = 15
    l_hx = 5
    w_hx = 6 * 4  # 2
    h_hx = 6 * 2
    l_down_duct = 10
    l_fan = 5
    d_fan = 5 * 2  # NILS: not used
    l_nozzle = 5
    d_nozzle = 4 * 2
    
    parametrise_axial_ducted_radiator(
        l,# w, h,
        w_intake, h_intake,
        l_up_duct,
        l_hx, w_hx, h_hx,
        l_down_duct,
        l_fan, d_fan,
        l_nozzle, d_nozzle,
    )

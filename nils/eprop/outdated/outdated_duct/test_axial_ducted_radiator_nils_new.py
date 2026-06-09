'''
Non-axisymmetric Powered Engine Nacelle (PEN)
'''
import os
import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt

from cst_modeling.io import output_surface
from cst_modeling.operation_new import Lofting_Revolution
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
        profile_x, profile_y = nacelle_profile.get_profile()
        
        # # after profile_x, profile_y = nacelle_profile.get_profile()
        # # enforce vertical half-height (h/2)
        # desired_half_height = h / 2.0
        
        # max_py = np.max(np.abs(profile_y))
        # if max_py > 0:
        #     profile_y_scaled = profile_y * (desired_half_height / max_py)
        # else:
        #     profile_y_scaled = profile_y.copy()
        # profile_y_scaled = profile_y.copy() * r_scale_array[i_sec]
        profile_y_scaled = profile_y
        
        profiles.append([profile_x, profile_y_scaled])
        # =============================================================================
        
        if i_sec == len(circum_control_psi) - 1:
            nacelle_profile.plot(show=True)
            
        print(max(profile_y_scaled))
    
    # sys.exit()
    
    #* Nacelle surface of revolution
    
    section_s_loc=[0.0, 0.25, 0.50, 0.75]
    
    # =============================================================================
    max_profile_y = max(np.max(np.abs(p[1])) for p in profiles)
    nacelle = Lofting_Revolution(
        profiles=profiles,
        section_s_loc=section_s_loc,  # [a/360.0 for a in circum_control_psi]
        section_x=0.0,
        # section_radius=0.25 * max_profile_y,  # 100,  # 0.0
        # section_radius=w / 2.0,
        section_radius=10.0,
        section_scale=1.0,
        n_spanwise=21,  # 51,
        section_shape='superellipse',   # <--- new
        superellipse_exp=5.0,           # <--- exponent = 5
        # =============================================================================
        y_scale=w/h,
        z_scale=1,
        # =============================================================================
    )
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

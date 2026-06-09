'''
Non-axisymmetric Powered Engine Nacelle (PEN)
'''
import os
import sys
sys.path.append(r'C:\Users\nmb48\Documents\GitHub\cst-modeling3d\nils\ducted_radiator')
# from guide_curve_getter import get_guide_curve
from guide_curve_getter_final import get_guide_curve
# from guide_curve_getter_final_equal_arc import get_guide_curve

import numpy as np
import matplotlib.pyplot as plt

from cst_modeling.io import output_surface
# from cst_modeling.operation_org import Lofting_Revolution
from cst_modeling.tools.axial_ducted_radiator_nils import NacelleIntakeHighlight, PoweredNacelleProfile


def parametrise_axial_ducted_radiator(
    l,
    w_intake, h_intake,
    l_up_duct,
    l_hx, w_hx, h_hx,
    l_down_duct,
    l_fan, d_fan,
    l_nozzle, d_nozzle,
):

    path = os.path.dirname(sys.argv[0])
    
    #* Nacelle Profile
    
    N = 100
    _, _, _, r_u_intake, r_l_intake, theta_u_intake, theta_l_intake = get_guide_curve(0.1, 0.1, w_intake, h_intake, N=N)
    _, _, _, r_u_hx, r_l_hx, theta_u_hx, theta_l_hx = get_guide_curve(0.1, 0.1, w_hx, h_hx, N=N)
    _, _, _, r_u_camber_bump, r_l_camber_bump, theta_u_camber_bump, theta_l_camber_bump = get_guide_curve(0.1, 0.1, 0.15, 0.06, N=N)
    _, _, _, r_u_te, r_l_te, theta_u_te, theta_l_te = get_guide_curve(0.1, 0.1, d_nozzle/2, d_nozzle/2, N=N)
    # equal_arc_length = True
    # _, _, _, r_u_intake, r_l_intake, theta_u_intake, theta_l_intake = get_guide_curve(0.1, 0.1, w_intake, h_intake, N=N, equal_arc_length=equal_arc_length)
    # _, _, _, r_u_hx, r_l_hx, theta_u_hx, theta_l_hx = get_guide_curve(0.1, 0.1, w_hx, h_hx, N=N, equal_arc_length=equal_arc_length)
    # _, _, _, r_u_camber_bump, r_l_camber_bump, theta_u_camber_bump, theta_l_camber_bump = get_guide_curve(0.1, 0.1, 0.15, 0.06, N=N, equal_arc_length=equal_arc_length)
    # _, _, _, r_u_te, r_l_te, theta_u_te, theta_l_te = get_guide_curve(0.1, 0.1, d_nozzle/2, d_nozzle/2, N=N, equal_arc_length=equal_arc_length)
    
    # =============================================================================
    #     r_intake_curve = np.hstack((r_u_intake[:-1], r_l_intake[:-1][::-1]))
    #     r_hx_curve = np.hstack((r_u_hx[:-1], r_l_hx[:-1][::-1]))
    #     r_camber_bump_curve = np.hstack((r_u_camber_bump[:-1], r_l_camber_bump[:-1][::-1]))
    #     r_te_curve = np.hstack((r_u_te[:-1], r_l_te[:-1][::-1]))
    #     
    #     theta_intake_curve = np.hstack((theta_u_intake[:-1], theta_l_intake[:-1][::-1]))
    #     theta_hx_curve = np.hstack((theta_u_hx[:-1], theta_l_hx[:-1][::-1]))
    #     theta_camber_bump_curve = np.hstack((theta_u_camber_bump[:-1], theta_l_camber_bump[:-1][::-1]))
    #     theta_te_curve = np.hstack((theta_u_te[:-1], theta_l_te[:-1][::-1]))
    
    r_intake_curve = np.hstack((r_u_intake, r_l_intake[:-1][::-1]))
    r_hx_curve = np.hstack((r_u_hx, r_l_hx[:-1][::-1]))
    r_camber_bump_curve = np.hstack((r_u_camber_bump, r_l_camber_bump[:-1][::-1]))
    r_te_curve = np.hstack((r_u_te, r_l_te[:-1][::-1]))
    
    theta_intake_curve = np.hstack((theta_u_intake, theta_l_intake[:-1][::-1]))
    theta_hx_curve = np.hstack((theta_u_hx, theta_l_hx[:-1][::-1]))
    theta_camber_bump_curve = np.hstack((theta_u_camber_bump, theta_l_camber_bump[:-1][::-1]))
    theta_te_curve = np.hstack((theta_u_te, theta_l_te[:-1][::-1]))
    # =============================================================================
    
    print('theta_intake_curve =', theta_intake_curve)
    # print('theta_hx_curve =', theta_hx_curve)
    # print('theta_camber_bump_curve =', theta_camber_bump_curve)
    # sys.exit()
    
    # # =============================================================================
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')  
    # # =============================================================================
    
    profiles = []
    
    # =============================================================================
    # minimal addition: record the per-section theta used for each saved profile
    section_theta_list = []
    # =============================================================================
    
    for i_sec in range(N):
        
        r_intake = r_intake_curve[i_sec]
        r_hx = r_hx_curve[i_sec]
        r_camber_bump = r_camber_bump_curve[i_sec]
        r_te = r_te_curve[i_sec]
        
        theta_intake = theta_intake_curve[i_sec]
        theta_hx = theta_hx_curve[i_sec]
        theta_camber_bump = theta_camber_bump_curve[i_sec]
        theta_te = theta_te_curve[i_sec]
    
        nacelle_profile = PoweredNacelleProfile(psi=None, n_point_segment=101)

        nacelle_profile.set_parameters(
            r_spinner=0.0, theta_spinner=90.0, r_fan=r_hx/2,
            highlight_x=-l_up_duct, highlight_y=r_intake/2,
            intake_face_center=(-l_up_duct, 0),
            # l_nacelle=l, r_te=d_nozzle/4,
            l_nacelle=l, r_te=r_te,
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
        
        profile_x, profile_y = nacelle_profile.get_profile()
        
        # profiles.append([profile_x, profile_y])
        
        # =============================================================================
        profiles.append([profile_x, profile_y])
        section_theta_list.append(theta_intake)    # <-- minimal: save angle used for this section
        # =============================================================================
        
        # nacelle_profile.plot(show=True)
        
        # print('np.max(profile_y) =', np.max(profile_y))
        
        # # =============================================================================
        # ax.scatter(profile_x, profile_y * np.cos(np.deg2rad(theta_intake)), profile_y * np.sin(np.deg2rad(theta_intake)), color='k', marker='.')
        # # ax.scatter(profile_x, profile_y * np.cos(np.deg2rad(theta_intake)), -profile_y * np.sin(np.deg2rad(theta_intake)), color='k', marker='.')
        # # =============================================================================
        
    # =============================================================================
    assert len(profiles) == len(section_theta_list), "profiles and section_theta_list must match length"
    # =============================================================================
    
    print('section_theta_list, np.diff(section_theta_list) =', section_theta_list, np.diff(section_theta_list))
    # sys.exit()
        
    # ax.set_aspect('equal')
    # plt.show()
    
    # =============================================================================
    # ======= Replace this block:
    # ax.set_aspect('equal')
    # plt.show()
    # sys.exit()
    #
    # With this minimal addition that builds a closed surface and writes the .dat
    # =======
    
    # =============================================================================
    #     ax.set_aspect('equal')
    #     plt.show()
    #     
    #     # --- Minimal additions: collect theta during the loop (see below) ---
    #     # (Assumes you appended theta_intake for each section into section_theta_list)
    #     # Build a closed surface: replicate the first section at the end for closure.
    #     n_sections = len(profiles)
    #     if n_sections == 0:
    #         raise RuntimeError("No profiles generated.")
    
    # ax.set_aspect('equal')
    # plt.show()
    
    # sys.exit()
    
    # --- Minimal addition: append the mirrored/reversed half so full-body is constructed ---
    # At this point: `profiles` has N entries and `section_theta_list` has N entries.
    # We append a reversed copy of the profiles and the negated reversed angles (mod 360).
    orig_profiles = profiles[:]                     # length N
    orig_thetas = np.array(section_theta_list)     # length N
    
    # =============================================================================
    #     # reversed list excluding the final duplicate that would equal the original first profile
    #     profiles_rev = orig_profiles[::-1][:-1]        # length N-1
    #     thetas_rev = ((-orig_thetas[::-1]) % 360.0)[:-1]  # length N-1
    #     
    #     # extend lists -> now represent full 360° sweep
    #     profiles.extend(profiles_rev)
    #     section_theta_list.extend(list(thetas_rev))
    
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
    # =============================================================================
    
    # Now continue with building the closed surface as before...
    n_sections = len(profiles)
    if n_sections == 0:
        raise RuntimeError("No profiles generated.")
    # =============================================================================
    
    # ensure we saved the per-section theta values during the loop
    try:
        section_theta_list  # (should have been appended inside the loop)
    except NameError:
        raise RuntimeError("You must collect per-section angles into 'section_theta_list' while building profiles.")
    
    n_point = profiles[0][0].shape[0]
    n_span = n_sections + 1  # duplicate first row at the end for closure
    
    surf_x = np.zeros((n_span, n_point))
    surf_y = np.zeros((n_span, n_point))
    surf_z = np.zeros((n_span, n_point))
    
    print('n_span =', n_span)
    for i_local in range(n_span):
        idx = i_local % n_sections              # wrap-around so last == first
        profile_x, profile_y = profiles[idx]
        # print('idx, len(section_theta_list) =', idx, len(section_theta_list))
        theta_deg = section_theta_list[idx]
        theta_rad = np.deg2rad(theta_deg)
    
        # Transform: X = axial (profile_x), Y = radial*cos(theta), Z = radial*sin(theta)
        surf_x[i_local, :] = profile_x
        surf_y[i_local, :] = profile_y * np.cos(theta_rad)
        surf_z[i_local, :] = profile_y * np.sin(theta_rad)
    
    # Put into the same structure your output_surface expects: list-of-surfaces
    surfs = [[surf_x, surf_y, surf_z]]
    
    # Write to .dat using your existing helper (ID=0 creates header; subsequent IDs append)
    for i_surf, surf in enumerate(surfs):
        output_surface(surf, fname=os.path.join(path, 'nacelle-non-axisymmetric.dat'), ID=i_surf)
    # =============================================================================
    
    # sys.exit()
    
    # # <<< TO BE REPLACED WITH CUSTOM LOFTING
    
    # # nacelle = Lofting_Revolution(
    # #     profiles=profiles,
    # #     section_s_loc=[a/360.0 for a in circum_control_psi],
    # #     section_x=0.0,
    # #     section_radius=0.0,
    # #     section_scale=1.0,
    # #     n_spanwise=51,
    # # )
    
    # # surfs = nacelle.sweep(interp_profile_kind='periodic')
    
    # # TO BE REPLACED WITH CUSTOM LOFTING >>>

    # for i_surf, surf in enumerate(surfs):
    #     output_surface(surf, fname=os.path.join(path, 'nacelle-non-axisymmetric.dat'), ID=i_surf)
    
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
        l,
        w_intake, h_intake,
        l_up_duct,
        l_hx, w_hx, h_hx,
        l_down_duct,
        l_fan, d_fan,
        l_nozzle, d_nozzle,
    )

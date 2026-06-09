"""
This script integrates the 3D parametric geometry of
the parallel electric ducted fan design with the two
separate heat exchanger nacelles.
"""

__all__ = ["integrate_outer_parallel_edf_axial_ducted_radiator"]

from nils.edf.outer_parallel.parallel_edf_geometry_generator import (
    generate_parallel_edf_geometry,
)
from nils.edf.outer_parallel.half_axial_ducted_radiator_geometry_generator import (
    generate_half_axial_ducted_radiator_geometry,
)

def integrate_outer_parallel_edf_axial_ducted_radiator(in_dict):
    
    in_dict_1 = in_dict["parallel_edf"]
    in_dict_2 = in_dict["axial_ducted_radiator"]

    #%% Generate separate meshes
    
    parallel_edf_mesh, x_dim_camber, y_dim_camber, profile_x_min, profile_x_maxx, outer_poly_pos_edf, outer_poly_neg_edf, core_poly_edf = generate_parallel_edf_geometry(
        in_dict_1['n_point_segment'],
        in_dict_1['r_hx_out'],
        in_dict_1['r_hx_in'],
        in_dict_1['r_hub_rotor'],
        in_dict_1['r_tip_rotor'],
        in_dict_1['l_rotor'],
        in_dict_1['l_stator'],
        in_dict_1['h_stator_in'],
        in_dict_1['h_stator_out'],
        in_dict_1['l_hx_side'],
        in_dict_1['alpha_incl'],
        in_dict_1['l_bp_nozzle'],
        in_dict_1['r_bp_nozzle_out'],
        in_dict_1['l_core_nozzle'],
        in_dict_1['r_core_nozzle_out'],
        in_dict_1['beta_core_nozzle'],
        in_dict_1['beta_bp_nozzle'],
        in_dict_1['l_spinner'],
        in_dict_1['l_intake'],
        in_dict_1['bypass_inner_angle'],
        in_dict_1['n_spanwise'],
        in_dict_1['save_dat'],
        in_dict_1['plot_profile'],
        in_dict_1['plot_mesh'],
        in_dict_1['f_scale'],
    )
    
    left_axial_ducted_radiator_mesh, _, left_outer_poly_pos_hx, left_outer_poly_neg_hx, left_hx_poly_pos, left_hx_poly_neg, left_hx_poly, left_fan_poly_pos, left_fan_poly_neg, left_fan_poly = generate_half_axial_ducted_radiator_geometry(
        in_dict_2['l'],
        in_dict_2['w_intake'], in_dict_2['h_intake'],
        in_dict_2['l_up_duct'],
        in_dict_2['l_hx'], in_dict_2['w_hx'], in_dict_2['h_hx'],
        in_dict_2['l_down_duct'],
        in_dict_2['l_fan'], in_dict_2['d_fan'],
        in_dict_2['l_nozzle'], in_dict_2['d_nozzle'],
        plot_mesh=False,
        dx=in_dict_2['dx'],
        dy=in_dict_2['dy'],
        dz=in_dict_2['dz'],
        camber_vert_tuple=(x_dim_camber, y_dim_camber),
    )
    
    right_axial_ducted_radiator_mesh, _, right_outer_poly_pos_hx, right_outer_poly_neg_hx, right_hx_poly_pos, right_hx_poly_neg, right_hx_poly, right_fan_poly_pos, right_fan_poly_neg, right_fan_poly = generate_half_axial_ducted_radiator_geometry(
        in_dict_2['l'],
        in_dict_2['w_intake'], in_dict_2['h_intake'],
        in_dict_2['l_up_duct'],
        in_dict_2['l_hx'], in_dict_2['w_hx'], in_dict_2['h_hx'],
        in_dict_2['l_down_duct'],
        in_dict_2['l_fan'], in_dict_2['d_fan'],
        in_dict_2['l_nozzle'], in_dict_2['d_nozzle'],
        plot_mesh=False,
        dx=in_dict_2['dx'],
        dy=in_dict_2['dy'],
        dz=in_dict_2['dz'],
        camber_vert_tuple=(x_dim_camber, -y_dim_camber),
        flip_z=True,
    )
    
    out_dict = {
        "parallel_edf": parallel_edf_mesh,
        "right_axial_ducted_radiator": right_axial_ducted_radiator_mesh,
        "left_axial_ducted_radiator": left_axial_ducted_radiator_mesh,
        "right_hx": right_hx_poly,
        "left_hx": left_hx_poly,
        "right_fan": right_fan_poly,
        "left_fan": left_fan_poly,
    }
    
    return out_dict






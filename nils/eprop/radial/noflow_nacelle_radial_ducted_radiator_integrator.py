__all__ = ["integrate_noflow_nacelle_radial_ducted_radiator"]

from nils.eprop.noflow_nacelle_geometry_generator import (
    generate_noflow_nacelle_geometry,
    convert_to_watertight_surface_mesh,
)
from nils.eprop.radial.radial_ducted_radiator_geometry_generator import (
    generate_radial_ducted_radiator_geometry,
)
from nils.eprop.actuator_disk_geometry_generator import (
    generate_actuator_disk_geometry,
)


def integrate_noflow_nacelle_radial_ducted_radiator(in_dict):
    
    in_dict_1 = in_dict["radial_ducted_radiator"]
    in_dict_2 = in_dict["noflow_nacelle"]
    in_dict_3 = in_dict["actuator_disk"]

    #%% Generate separate meshes
    
    radial_ducted_radiator_mesh, outer_duct_surface_mesh, inner_duct_surface_mesh = generate_radial_ducted_radiator_geometry(
        # Upstream duct
        AR=in_dict_1['AR'],
        # A_out=A_out,
        # r_out=r_out,
        delta_x=in_dict_1['delta_x'],
        delta_r=in_dict_1['delta_r'],
        # HX
        l_hx=in_dict_1['l_hx'],
        r_hx_out=in_dict_1['r_hx_out'],
        r_hx_in=in_dict_1['r_hx_in'],
        # Gaussian bump
        h_bump=in_dict_1['h_bump'],
        # Downstream duct
        l_down_duct=in_dict_1['l_down_duct'],
        # Fan
        dz_hx_fan=in_dict_1['dz_hx_fan'],
        d_fan=in_dict_1['d_fan'],
        l_fan=in_dict_1['l_fan'],
        # Nozzle
        d_nozzle=in_dict_1['d_nozzle'],
        l_nozzle=in_dict_1['l_nozzle'],
        
        N_up_duct_slices=in_dict_1['N_up_duct_slices'],
        N_bump_slices=in_dict_1['N_bump_slices'],
        
        n_outer_surf_sec=in_dict_1['n_outer_surf_sec'],
        n_inner_surf_sec=in_dict_1['n_inner_surf_sec'],
        n_hx_surf_sec=in_dict_1['n_hx_surf_sec'],
        nn_sect=in_dict_1['nn_sect'],
        nn_surf=in_dict_1['nn_surf'],
        ns_surf=in_dict_1['ns_surf'],
        
        plot_mesh=in_dict_1['plot_mesh'],
        save_dat=in_dict_1['save_dat'],
        
        dx=in_dict_1['dx'],
        dy=in_dict_1['dy'],
        dz=in_dict_1['dz'],
    )
    
    res = generate_noflow_nacelle_geometry(
        in_dict_2['l_tot'], in_dict_2['w_tot'], in_dict_2['h_tot'],
        in_dict_2['w_le'], in_dict_2['w_te'], in_dict_2['beta_horz'],
        in_dict_2['h_le'], in_dict_2['h_te'], in_dict_2['beta_vert'],
        N_u=in_dict_2['N_u'], N_l=in_dict_2['N_l'],
        camber_vert_tuple=in_dict_2['camber_vert_tuple'],
        N_psi=in_dict_2['N_psi'], N_eta=in_dict_2['N_eta'],
        plot=False,
        write_dat_file=False,
    )
    noflow_nacelle_mesh = convert_to_watertight_surface_mesh(res)
    spinner_mesh = res['spinner_dict']['vtk_dataset']
    actuator_disk_mesh = generate_actuator_disk_geometry(
        in_dict_3['r_outer'], in_dict_3['r_inner'], in_dict_3['x_loc'], in_dict_3['y_loc'], in_dict_3['z_loc'],
    )
    
    out_dict = {
        "radial_ducted_radiator": radial_ducted_radiator_mesh,
        "noflow_nacelle": noflow_nacelle_mesh,
        "spinner": spinner_mesh,
        "actuator_disk": actuator_disk_mesh,
    }
    
    return out_dict



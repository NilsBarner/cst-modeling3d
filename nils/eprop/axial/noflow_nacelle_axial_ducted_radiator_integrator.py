__all__ = ["integrate_noflow_nacelle_axial_ducted_radiator"]

from nils.eprop.noflow_nacelle_geometry_generator import (
    generate_noflow_nacelle_geometry,
    convert_to_watertight_surface_mesh,
)
from nils.eprop.axial.axial_ducted_radiator_geometry_generator import (
    generate_axial_ducted_radiator_geometry,
)
from nils.eprop.fairing_geometry_generator import generate_fairing_geometry
from nils.eprop.actuator_disk_geometry_generator import (
    generate_actuator_disk_geometry,
)


def integrate_noflow_nacelle_axial_ducted_radiator(in_dict):
    
    in_dict_1 = in_dict["axial_ducted_radiator"]
    in_dict_2 = in_dict["noflow_nacelle"]
    in_dict_3 = in_dict["fairing"]
    in_dict_4 = in_dict["actuator_disk"]

    #%% Generate separate meshes
    
    axial_ducted_radiator_mesh, hx_mesh_pos, hx_mesh_neg, hx_mesh, fan_mesh = generate_axial_ducted_radiator_geometry(
        in_dict_1['l'],
        in_dict_1['w_intake'], in_dict_1['h_intake'],
        in_dict_1['l_up_duct'],
        in_dict_1['l_hx'], in_dict_1['w_hx'], in_dict_1['h_hx'],
        in_dict_1['l_down_duct'],
        in_dict_1['l_fan'], in_dict_1['d_fan'],
        in_dict_1['l_nozzle'], in_dict_1['d_nozzle'],
        plot_profile=True,
        plot_mesh=False,
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
    
    fairing_mesh = generate_fairing_geometry(
        in_dict_3['l_fairing'],
        in_dict_3['t_fairing'],
        in_dict_3['h_fairing'],
        in_dict_3['dx_fairing'],
        in_dict_3['dy_fairing'],
        in_dict_3['dz_fairing'],
    )
    
    actuator_disk_mesh = generate_actuator_disk_geometry(
        in_dict_4['r_outer'], in_dict_4['r_inner'], in_dict_4['x_loc'], in_dict_4['y_loc'], in_dict_4['z_loc'],
    )
    
    out_dict = {
        "axial_ducted_radiator": axial_ducted_radiator_mesh,
        "hx": hx_mesh,
        "fan": fan_mesh,
        "noflow_nacelle": noflow_nacelle_mesh,
        "spinner": spinner_mesh,
        "fairing": fairing_mesh,
        "actuator_disk": actuator_disk_mesh,
    }
    
    return out_dict


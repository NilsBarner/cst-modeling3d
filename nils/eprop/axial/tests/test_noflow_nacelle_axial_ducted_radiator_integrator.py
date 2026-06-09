__all__ = []

import numpy as np

from nils.eprop.axial.noflow_nacelle_axial_ducted_radiator_integrator import (
    integrate_noflow_nacelle_axial_ducted_radiator,
)
from nils.eprop.axial.noflow_nacelle_axial_ducted_radiator_plotter import (
    plot_noflow_nacelle_axial_ducted_radiator,
)

in_dict = {}

#%% No-flow nacelle inputs

in_dict_2 = in_dict["noflow_nacelle"] = {}

f_scale_prop_nac = 2.5

# Discretisation
in_dict_2['N_psi'] = 200
in_dict_2['N_eta'] = 200

# Overall dimensions
in_dict_2['l_tot'] = 4.0 * f_scale_prop_nac
in_dict_2['w_tot'] = 1.2 * 0.8 * f_scale_prop_nac
in_dict_2['h_tot'] = 1.2 * 1.2 * f_scale_prop_nac

# Leading edge
in_dict_2['w_le'] = 0.1 * 2 * f_scale_prop_nac
in_dict_2['h_le'] = 0.1 * 3 * f_scale_prop_nac

# Trailing edge
in_dict_2['w_te'] = 0.15 * 1 * f_scale_prop_nac
in_dict_2['h_te'] = 0.1 * 3 * f_scale_prop_nac
in_dict_2['beta_horz'] = np.deg2rad(8.0)
in_dict_2['beta_vert'] = np.deg2rad(6.0)

# Upper half
in_dict_2['N_u'] = 0.45

# Lower half
in_dict_2['N_l'] = 0.25

# Vertical camber
in_dict_2['camber_vert_tuple'] = (0.0, 0.35)  # (-0.01 * 4, 0.35)  # None

#%% Axial ducted radiator inputs

in_dict_1 = in_dict["axial_ducted_radiator"] = {}

f_scale = 2

in_dict_1['l'] = 3.2 * f_scale_prop_nac
in_dict_1['w'] = 1.2 * f_scale * f_scale_prop_nac
in_dict_1['h'] = 0.6 * f_scale * f_scale_prop_nac
in_dict_1['w_intake'] = 0.8 * f_scale * f_scale_prop_nac
in_dict_1['h_intake'] = 0.3 * f_scale * f_scale_prop_nac
in_dict_1['l_up_duct'] = 1.5 * f_scale_prop_nac
in_dict_1['l_hx'] = 0.3 * f_scale_prop_nac
in_dict_1['w_hx'] = 1.2 * f_scale * f_scale_prop_nac
in_dict_1['h_hx'] = 0.6 * f_scale * f_scale_prop_nac
in_dict_1['l_down_duct'] = 0.8 * f_scale_prop_nac
in_dict_1['l_fan'] = 0.3 * f_scale_prop_nac
in_dict_1['d_fan'] = 0.5 * f_scale_prop_nac
in_dict_1['l_nozzle'] = 0.5 * f_scale_prop_nac
in_dict_1['d_nozzle'] = 0.3 * f_scale * f_scale_prop_nac

#%% Actuator disk geometry inputs

in_dict_4 = in_dict["actuator_disk"] = {}

in_dict_4['r_outer'] = 3.93
in_dict_4['r_inner'] = 0.1
in_dict_4['x_loc'] = 0.0
in_dict_4['y_loc'] = 0.0
in_dict_4['z_loc'] = 0.0

#%% Fairing inputs

in_dict_3 = in_dict["fairing"] = {}

in_dict_3['l_fairing'] = 1.0 * f_scale_prop_nac
in_dict_3['t_fairing'] = 0.15 * in_dict_3['l_fairing'] * f_scale_prop_nac
in_dict_3['h_fairing'] = 0.2 * f_scale_prop_nac
in_dict_3['dx_fairing'] = 1.2 * f_scale_prop_nac
in_dict_3['dy_fairing'] = 0.0
in_dict_3['dz_fairing'] = -2.3

#%% Integration inputs

in_dict_1['dx'] = 5
in_dict_1['dy'] = 0.0
in_dict_1['dz'] = -(in_dict_2['h_tot'] / 2 + in_dict_1['h_hx'] / 2)

#%%

out_dict = integrate_noflow_nacelle_axial_ducted_radiator(in_dict)

#%%

plot_noflow_nacelle_axial_ducted_radiator(out_dict)



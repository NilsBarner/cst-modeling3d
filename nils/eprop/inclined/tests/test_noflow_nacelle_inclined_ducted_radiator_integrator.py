__all__ = []

import numpy as np

from nils.eprop.inclined.noflow_nacelle_inclined_ducted_radiator_integrator import (
    integrate_noflow_nacelle_inclined_ducted_radiator,
)
from nils.eprop.inclined.noflow_nacelle_inclined_ducted_radiator_plotter import (
    plot_noflow_nacelle_inclined_ducted_radiator,
)

in_dict = {}

#%% No-flow nacelle inputs

in_dict_2 = in_dict["noflow_nacelle"] = {}

f_scale = 2.5

# Discretisation
in_dict_2['N_psi'] = 200
in_dict_2['N_eta'] = 200

# Overall dimensions
in_dict_2['l_tot'] = 4.0 * f_scale
in_dict_2['w_tot'] = 1.2 * 0.8 * f_scale
in_dict_2['h_tot'] = 1.2 * 1.2 * f_scale

# Leading edge
in_dict_2['w_le'] = 0.1 * 2 * f_scale
in_dict_2['h_le'] = 0.1 * 3 * f_scale

# Trailing edge
# in_dict_2['w_te'] = 0.15 * 2 * f_scale
in_dict_2['h_te'] = 0.1 * 3 * f_scale
in_dict_2['beta_horz'] = np.deg2rad(8.0)
in_dict_2['beta_vert'] = np.deg2rad(6.0)

# Upper half
in_dict_2['N_u'] = 0.45

# Lower half
in_dict_2['N_l'] = 0.25

# Vertical camber
in_dict_2['camber_vert_tuple'] = (0.0, 0.35)  # (-0.01 * 4, 0.35)  # None

#%% Inclined ducted radiator inputs

in_dict_1 = in_dict["inclined_ducted_radiator"] = {}

# Intake
in_dict_1['dz_hx_intake'] = -0.5
in_dict_1['w_intake'] = 0.75
in_dict_1['h_intake'] = 0.25
# Upstream duct
in_dict_1['l_up_duct'] = 4
# HX
in_dict_1['l_hx'] = 1.5
in_dict_1['w_hx'] = 1.5
in_dict_1['h_hx'] = 0.75
in_dict_1['dx_hx_corner'] = 0.5
in_dict_1['dz_hx_corner'] = 0.15
in_dict_1['alpha'] = 22.78
in_dict_1['beta'] = 115.23
# Downstream duct
# in_dict_1['l_down_duct'] = 2
# Fan
in_dict_1['dz_hx_fan'] = 0.5
in_dict_1['d_fan'] = 0.5
in_dict_1['l_fan'] = 0.3
# Nozzle
in_dict_1['d_nozzle'] = 0.3
in_dict_1['l_nozzle'] = 0.3

# =============================================================================
in_dict_1['l_down_duct'] = in_dict_2['l_tot'] - (in_dict_1['l_nozzle'] + in_dict_1['l_fan'] + in_dict_1['l_hx'] + in_dict_1['l_up_duct'])
in_dict_2['w_te'] = in_dict_1['d_fan'] / 2
# =============================================================================

in_dict_1['N_inlet_up'] = 0.25
in_dict_1['N_outlet_up'] = 0.005
in_dict_1['n_slices_up'] = 10

in_dict_1['N_inlet_down'] = 0.005
in_dict_1['N_outlet_down'] = 0.5  # 0.25
in_dict_1['n_slices_down'] = 10

in_dict_1['n_surf_sec'] = in_dict_1['n_slices_up'] + in_dict_1['n_slices_down'] + 5
in_dict_1['nn_sect'] = 500
in_dict_1['nn_surf'] = 201
in_dict_1['ns_surf'] = 51

in_dict_1['plot_sections'] = False
in_dict_1['plot_scatter'] = False
in_dict_1['plot_mesh'] = False
in_dict_1['save_dat'] = False

#%% Actuator disk geometry inputs

in_dict_4 = in_dict["actuator_disk"] = {}

in_dict_4['r_outer'] = 3.93
in_dict_4['r_inner'] = 0.1
in_dict_4['x_loc'] = 0.0
in_dict_4['y_loc'] = 0.0
in_dict_4['z_loc'] = 0.0

#%% Integration inputs

in_dict_1['dx'] = in_dict_1['l_hx'] / 2 + in_dict_1['l_up_duct']  # 4.0  # 0.0
in_dict_1['dy'] = 0.0
in_dict_1['dz'] = -0.5

#%%

out_dict = integrate_noflow_nacelle_inclined_ducted_radiator(in_dict)

#%%

plot_noflow_nacelle_inclined_ducted_radiator(out_dict)











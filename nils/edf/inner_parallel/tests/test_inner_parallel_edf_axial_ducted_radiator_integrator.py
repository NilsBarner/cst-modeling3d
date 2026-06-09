__all__ = []

import numpy as np

from nils.edf.inner_parallel.inner_parallel_edf_axial_ducted_radiator_integrator import (
    integrate_inner_parallel_edf_axial_ducted_radiator,
)
from nils.edf.inner_parallel.inner_parallel_edf_axial_ducted_radiator_plotter import (
    plot_inner_parallel_edf_axial_ducted_radiator,
)

in_dict = {}

#%% Parallel edf inputs

in_dict_1 = in_dict["parallel_edf"] = {}

_ = 15

in_dict_1['n_point_segment'] = 101
in_dict_1['r_hx_out'] = 20.5 / _
in_dict_1['r_hx_in'] = 14 / _
in_dict_1['r_hub_rotor'] = 6 / _
in_dict_1['r_tip_rotor'] = 20 / _
in_dict_1['l_rotor'] = 4 / _
in_dict_1['l_stator'] = 2 / _
in_dict_1['h_stator_in'] = 6 / _
in_dict_1['h_stator_out'] = 8 / _
in_dict_1['l_hx_side'] = 0 / _
# in_dict_1['alpha_incl'] = np.deg2rad(35)
in_dict_1['alpha_incl'] = np.deg2rad(90)
in_dict_1['l_bp_nozzle'] = 18 / _
in_dict_1['r_bp_nozzle_out'] = 19 / _
in_dict_1['l_core_nozzle'] = 25 / _
in_dict_1['r_core_nozzle_out'] = 9 / _
in_dict_1['beta_core_nozzle'] = -np.deg2rad(10)
in_dict_1['beta_bp_nozzle'] = -np.deg2rad(10)
in_dict_1['l_spinner'] = 6 / _
in_dict_1['l_intake'] = 20 / _
in_dict_1['bypass_inner_angle'] = 0.0
in_dict_1['n_spanwise'] = 21
in_dict_1['save_dat'] = False
in_dict_1['plot_profile'] = True
in_dict_1['plot_mesh'] = False
in_dict_1['f_scale'] = 1 / _

#%% Axial ducted radiator inputs

in_dict_2 = in_dict["axial_ducted_radiator"] = {}

in_dict_2['l'] = 3.2
in_dict_2['w'] = 1.2
in_dict_2['h'] = 0.6
in_dict_2['w_intake'] = 0.6
in_dict_2['h_intake'] = 0.3
in_dict_2['l_up_duct'] = 1.5
in_dict_2['l_hx'] = 0.3
in_dict_2['w_hx'] = 1.2
in_dict_2['h_hx'] = 0.6
in_dict_2['l_down_duct'] = 0.8
in_dict_2['l_fan'] = 0.3
in_dict_2['d_fan'] = 0.5
in_dict_2['l_nozzle'] = 0.3
in_dict_2['d_nozzle'] = 0.3

#%% Integration inputs

in_dict_2['dx'] = 0.0
in_dict_2['dy'] = 0.0
in_dict_2['dz'] = 0.0

in_dict_2['n_spanwise'] = 21

#%%

out_dict = integrate_inner_parallel_edf_axial_ducted_radiator(in_dict)

#%%

plot_inner_parallel_edf_axial_ducted_radiator(out_dict)



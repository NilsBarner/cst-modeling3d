__all__ = []

import numpy as np

from nils.edf.series.series_edf_geometry_generator import (
    generate_series_edf_geometry,
)
from nils.edf.series.series_edf_plotter import plot_series_edf


in_dict = {}

in_dict['f_scale'] = 1.0

in_dict['n_point_segment'] = 101
in_dict['r_hx_out'] = 20.5 * in_dict['f_scale']
in_dict['r_hx_in'] = 14 * in_dict['f_scale']
in_dict['r_hub_rotor'] = 6 * in_dict['f_scale']
in_dict['r_tip_rotor'] = 20 * in_dict['f_scale']
in_dict['l_rotor'] = 4 * in_dict['f_scale']
in_dict['l_stator'] = 3 * in_dict['f_scale']
in_dict['h_stator_in'] = 6 * in_dict['f_scale']
in_dict['h_stator_out'] = 8 * in_dict['f_scale']
in_dict['l_hx_side'] = 8 * in_dict['f_scale']
# in_dict['alpha_incl'] = np.deg2rad(35)
in_dict['alpha_incl'] = np.deg2rad(90)
in_dict['l_bp_nozzle'] = 18 * in_dict['f_scale']
in_dict['r_bp_nozzle_out'] = 19 * in_dict['f_scale']
in_dict['l_core_nozzle'] = 25 * in_dict['f_scale']
in_dict['r_core_nozzle_out'] = 9 * in_dict['f_scale']
in_dict['beta_core_nozzle'] = -np.deg2rad(10)
in_dict['beta_bp_nozzle'] = -np.deg2rad(10)
in_dict['l_spinner'] = 6 * in_dict['f_scale']
in_dict['l_intake'] = 20 * in_dict['f_scale']
in_dict['bypass_inner_angle'] = 0.0
in_dict['n_spanwise'] = 21
in_dict['save_dat'] = False
in_dict['plot_profile'] = True
in_dict['plot_mesh'] = False
in_dict['dx'] = 0.0
in_dict['dy'] = 0.0
in_dict['dz'] = 0.0

#%%

out_dict = generate_series_edf_geometry(
    in_dict,
)

#%%

plot_series_edf(out_dict)


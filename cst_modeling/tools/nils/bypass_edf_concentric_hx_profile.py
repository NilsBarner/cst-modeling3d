"""
This script features plotting methods based on Kulfan's shape space
transforms, is based on \cst-modeling3d\cst_modeling\tools\nacelle.py.
It supersedes edf_bypass_model_cst_backup.py.
"""

__all__ = [
    "NacelleIntakeHighlight",
    "PoweredNacelleProfile",
]

'''
Nacelle profiles for lofting to a surface of revolution.

- flow through nacelle (FTN);
- powered engine nacelle (PEN), or with powered nacelle (WPN);


References:

    [1] "Comparison of Overwing and Underwing Nacelle Aeropropulsion Optimization for Subsonic Transport Aircraft", Journal of Aircraft, 2024, Vol. 61, No. 2.
        https://doi.org/10.2514/1.C037508
        
    [2] "Non-axisymmetric aero-engine nacelle design by surrogate-based methods", Aerospace Science and Technology, 2021, Vol. 117, 106890.
        https://doi.org/10.1016/j.ast.2021.106890
        
    [3] "Impact of Droop and Scarf on the Aerodynamic Performance of Compact Aero-Engine Nacelles", AIAA SciTech, 2020.
        https://doi.org/10.2514/6.2020-1522

'''
import copy
import numpy as np

from typing import Dict, Tuple, List, Union
from matplotlib.collections import LineCollection
from scipy.interpolate import interp1d, CubicSpline

from cst_modeling.math import rotate_vector, transform, cst_foil, dist_clustcos, interp_from_curve
from cst_modeling.tools.auxiliary import section_flap
from cst_modeling.section import Section
from cst_modeling.basic import transform
from cst_modeling.math import intersect_index, interp_from_curve

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from scipy.interpolate import CubicSpline
from scipy.optimize import fsolve

from flydrogen.systems.edf.cst_plotting.streamlined_body_generator import generate_streamlined_body_geometry, find_beta_threshold


class NacelleIntakeHighlight():
    '''
    The three-dimensional highlight point curve of the intake face.
    
    Parameters
    ----------
    offset_y_intake : float
        Offset of the intake face center from the axis of rotation.
        
    l_intake : float
        Offset of the intake face center from the fan face, i.e., the intake length.
        
    theta_droop : float
        Centre-line droop angle (degree), i.e., arctan(offset_y_intake/l_intake).
        After the revolution around the engine axis, the contoured intake was morphed by the droop angle, 
        measured as the offset between the engine and the inlet centre-line.
        
    theta_scarf : float
        Highlight scarf angle (degree).
        The geometry was scarfed by rotating the highlight plane around the intake face center.
        
    n_circum : int
        The number of circumferential points for the intake highlight curve.
        
    circum_control_psi : List[float]
        Circumferential location of control sections.
        
    circum_control_r_highlight : Union[float, List[float]]
        Radius of the intake highlight point to the intake face center.
        If float, the radius is constant for all sections.
    
    '''
    def __init__(self, l_intake: float, 
                    theta_droop: float, theta_scarf: float,
                    n_circum=101, 
                    circum_control_psi = [0.0, 90.0, 180.0, 270.0],
                    circum_control_r_highlight : Union[float, List[float]] = 1.0,
                ) -> None:
        
        if theta_droop + theta_scarf > 45.0:
            raise ValueError('The sum of theta_droop and theta_scarf should be less than 45.0 degree.')
        
        self.l_intake = l_intake
        self.theta_droop = theta_droop
        self.theta_scarf = theta_scarf
        self.n_circum = n_circum
        self.circum_control_psi = circum_control_psi + [360.0]
        
        if isinstance(circum_control_r_highlight, float):
            self.circum_control_r_highlight = [circum_control_r_highlight] * len(self.circum_control_psi)
        else:
            self.circum_control_r_highlight = circum_control_r_highlight + circum_control_r_highlight[0]
        
        self.intake_face_center = np.zeros(3)
        self.intake_face_center[0] = -l_intake
        self.intake_face_center[1] =  l_intake * np.tan(np.deg2rad(theta_droop))
        
    def calculate(self) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Calculate highlight point curve of the intake.
        
        Returns
        -------
        curve : np.ndarray [n_n_circum, 3]
            Highlight point coordinates in 3D space, i.e., (x, y, z).
            
        psi_curve: np.ndarray [n_n_circum, 4]
            Highlight point coordinates in 3D space, described in parametric coordinates,
            i.e., (psi, x, y, z).
        '''
        
        self.func_radius = CubicSpline(self.circum_control_psi, self.circum_control_r_highlight, bc_type='periodic')
        
        curve_psi = np.linspace(0.0, 360.0, self.n_circum, endpoint=True)
        
        curve_radius = self.func_radius(curve_psi)
        
        #* The intake highlight curve in the yz-plane before rotation
        
        curve = np.zeros((self.n_circum, 3))
        curve[:, 0] = self.intake_face_center[0] / np.cos(np.deg2rad(self.theta_droop))
        curve[:, 1] = curve_radius * np.cos(np.deg2rad(curve_psi))
        curve[:, 2] = curve_radius * np.sin(np.deg2rad(curve_psi))
        
        
        #* Rotate yz-plane about z-axis, the origin is the fan face center
        #  After the revolution around the engine axis, the contoured intake was morphed by the droop angle, 
        #  measured as the offset between the engine and the inlet centre-line.
        
        curve = rotate_vector(curve[:, 0], curve[:, 1], curve[:, 2], angle=self.theta_droop,
                       origin=[0.0, 0.0, 0.0], axis_vector=[0.0, 0.0, 1.0])
        
        #* Rotate xyz about z-axis, the origin is the intake face center
        #  The geometry was scarfed by rotating the highlight plane around the intake face center.
        
        curve = rotate_vector(curve[:, 0], curve[:, 1], curve[:, 2], angle=self.theta_scarf,
                       origin=self.intake_face_center, axis_vector=[0.0, 0.0, 1.0])
                
        avg = 0.5*(curve[0] + curve[-1])
        for i in range(avg.shape[0]):
            if abs(avg[i]) < 1e-6:
                avg[i] = 0.0
        
        curve[ 0,:] = avg
        curve[-1,:] = avg
           
        #* Project the highlight point to the profile plane
        
        new_curve_psi = [np.arctan2(z, y) for y, z in zip(curve[:self.n_circum, 1], curve[:self.n_circum, 2])]

        for i in range(len(new_curve_psi)):
            if new_curve_psi[i] < 0.0:
                new_curve_psi[i] += 2.0 * np.pi

        new_curve_psi[-1] = 2.0 * np.pi

        self.func_x = CubicSpline(new_curve_psi, curve[:,0], bc_type='periodic')
        self.func_y = CubicSpline(new_curve_psi, curve[:,1], bc_type='periodic')
        self.func_z = CubicSpline(new_curve_psi, curve[:,2], bc_type='periodic')
        
        psi_curve = np.zeros((self.n_circum, 4))
        psi_curve[:, 0] = curve_psi
        psi_curve[:, 1] = self.func_x(new_curve_psi)
        psi_curve[:, 2] = self.func_y(new_curve_psi)
        psi_curve[:, 3] = self.func_z(new_curve_psi)
                
        return curve, psi_curve

    def get_coordinate_3d(self, psi: float) -> np.ndarray:
        '''
        Get the highlight point coordinate at a given circumferential angle.
        
        Parameters
        ----------
        psi : float
            Circumferential angle (degree).
            
        Returns
        -------
        point : np.ndarray [3]
            Highlight point coordinate in 3D space, i.e., (x, y, z).
        '''
        psi = np.deg2rad(psi % 360.0)
        
        x = self.func_x(psi)
        y = self.func_y(psi)
        z = self.func_z(psi)
        
        return np.array([x, y, z])
    
    def get_coordinate_2d(self, psi: float) -> np.ndarray:
        '''
        Get the highlight point coordinate at a given circumferential angle.
        
        Parameters
        ----------
        psi : float
            Circumferential angle (degree).
            
        Returns
        -------
        point : np.ndarray [3]
            Highlight point coordinate in 3D space, i.e., (x, y, z).
        '''
        point = self.get_coordinate_3d(psi)
        
        y = np.sqrt(point[1]**2 + point[2]**2)
        
        return np.array([point[0], y])


class PoweredNacelleProfile():
    '''
    Two-dimensional profile of a powered engine nacelle.
    
    The reference point (0,0) is located at the intersection of rotation axis and the fan inlet face. 
    The definition of the profile parameters can be found in Figure 1 of [2].
    
    The intake face center is the same for all profiles in different meridional angles,
    which locates at (-l_intake, offset_y_intake, 0.0).
    
    
    Nacelle profile consists of the following curves:
    
    - Outer cowl surface: (3,4);
    - Inlet surface: (3,2);
    - Bypass duct outer and inner surfaces: (5,4), (6,7);
    - Core duct outer and inner surfaces: (9,8), (10,11);
    
    - Spinner (conical surface): (0,1);
    - Core cowl (conical surface): (7, 8);
    - Core plug (conical surface): (11,12);
    
    - Fan face intake (planar surface): (1,2);
    - Bypass duct inlet (planar surface): (5,6);
    - Core duct inlet (planar surface): (9,10);
    
    
    Nacelle profile has the following features:
    
    - l_fore: fore-body length;
    - l_lip: lip length;
    
    - r_if: initial fore-body radius;
    - r_throat: inlet throat radius;
    - r_max: maximum radius;

    - theta_airfoil: airfoil geometric incidence angle (degree);
    - theta_bp = bypass exit angle (degree);
    - theta_cr = core exit angle (degree);
    - theta_nac = nacelle boat tail angle (degree);
    
    - x_max: Axial location of maximum radius;
    - x_thr: Axial location of inlet throat;
    
    Parameters
    ----------
    psi : float
        Profile's meridional angle (degree), [0, 360].
        Zero angle corresponds to the half x-y plane with positive y.
    
    n_point_segment : int, optional
        Number of points for each segment of the profile. The default is 201.
    
    Attributes
    ----------
    params : Dict[str, float]
        Nacelle profile parameters.
        
    segment_points : Dict[int, np.ndarray]
        Nacelle profile segment points.
        
    profile_segments : Dict[str, np.ndarray]
        Nacelle profile segments.
    
    features : Dict[str, float]
        Nacelle profile features.
        
    profile_x, profile_y : np.ndarray [n_point_profile]
        Nacelle profile coordinates.
    
    '''

    def __init__(self, psi=0.0, n_point_segment=201) -> None:
        
        self.psi = psi
        self.n_point_segment = n_point_segment
        
        self.params : Dict[str, np.ndarray] = {}
        self.features : Dict[str, np.ndarray] = {}
        self.segment_points : Dict[int, np.ndarray] = {}
        self.profile_segments : Dict[int, np.ndarray] = {}
        
        self.profile_x : np.ndarray = None
        self.profile_y : np.ndarray = None
        
        
    def set_parameters(self,
        r_hx_out: float,
        r_hx_in: float,
        r_hub_rotor: float,
        r_tip_rotor: float,
        l_rotor: float,
        l_stator: float,
        h_stator_in: float,
        h_stator_out: float,
        l_hx_side: float,
        alpha_incl: float,
        l_bp_nozzle: float,
        r_bp_nozzle_out: float,
        l_core_nozzle: float,
        r_core_nozzle_out: float,
        beta_core_nozzle: float,
        beta_bp_nozzle: float,
        l_spinner: float,
        l_intake: float,
        r_in_spiral: float,
        t_out_spiral: float,
        N_spirals_t: float,
        cst_u: List[float], cst_l: List[float],
        bypass_inner_angle: float = 0.0,
        bypass_inner_control_points: List[Tuple[float, float]] = None,
        core_outer_control_points: List[Tuple[float, float]] = None,
        core_inner_control_points: List[Tuple[float, float]] = None,
    ) -> None:
        '''
        Set profile parameters.
        
        Parameters
        ----------
        cst_u, cst_l : List[float]
            Upper and lower CST coefficients of the airfoil profile.
        
        bypass_inner_angle : float
            Bypass duct inner surface half-angle (degree) at point (6).
        
        bypass_inner_control_points : List[Tuple[float, float]], optional
            Control points for the bypass duct inner surface. The default is None.
            
        core_outer_control_points : List[Tuple[float, float]], optional
            Control points for the core duct outer surface. The default is None.
            
        core_inner_control_points : List[Tuple[float, float]], optional
            Control points for the core duct inner surface. The default is None.
        
        Profile parameters:
        
        - theta_spinner: half-angle of the spinner cone (degree) (0);
        
        - r_spinner: spinner radius (1);
        - r_fan: fan radius (optional) (2);
        
        - intake_face_center: intake face center, ndarray [3];
        - highlight_x: highlight point x-coordinate (3);
        - highlight_y: highlight point y-coordinate (3);
        
        - l_nacelle: nacelle length in the x-axis direction (4);
        - r_te: nacelle trailing edge radius (4);
        
        - l_fan: fan length in the x-axis direction (5, 6);
        - r_bypass_outer: bypass duct outer radius (5);
        - r_bypass_inner: bypass duct inner radius (6);

        - x_core_cowl_0: Axial location of core cowl start (7);
        - y_core_cowl_0: Radial location of core cowl start (7);
        - x_core_cowl_1: Axial location of core cowl end (8);
        - y_core_cowl_1: Radial location of core cowl end (8);
        
        - x_core_duct: Axial location of core duct inlet (9);
        - r_core_outer: core duct outer radius (9);
        - r_core_inner: core duct inner radius (10);
        
        - x_core_plug_0: Axial location of core plug start (11);
        - y_core_plug_0: Radial location of core plug start (11);
        - x_core_plug_1: Axial location of core plug end (12);
        
        '''
        
        self.params['cst_u'] = cst_u
        self.params['cst_l'] = cst_l
        self.params['r_hx_out'] = r_hx_out
        self.params['r_hx_in'] = r_hx_in
        self.params['r_hub_rotor'] = r_hub_rotor
        self.params['r_tip_rotor'] = r_tip_rotor
        self.params['l_rotor'] = l_rotor
        self.params['l_stator'] = l_stator
        self.params['h_stator_in'] = h_stator_in
        self.params['h_stator_out'] = h_stator_out
        # Inclined HX profile geometry
        '''   <------> l_hx_side
              ------- <
             /  /  /  |
            /  /  /   |
           /  /  /    | h_hx
          /  /  /     |
         /  /  / α    | 
        -------       >
            <-> l_hx_avg  
        '''
        self.params['l_hx_side'] = l_hx_side
        h_hx = self.params['r_hx_out'] - self.params['r_hx_in']
        self.params['h_hx'] = h_hx
        l_hx_avg = h_hx / np.tan(alpha_incl)
        self.params['l_hx_avg'] = l_hx_avg
        self.params['l_bp_nozzle'] = l_bp_nozzle
        self.params['r_bp_nozzle_out'] = r_bp_nozzle_out
        self.params['l_core_nozzle'] = l_core_nozzle
        self.params['r_core_nozzle_out'] = r_core_nozzle_out
        self.params['beta_core_nozzle'] = beta_core_nozzle
        self.params['beta_bp_nozzle'] = beta_bp_nozzle
        self.params['l_spinner'] = l_spinner
        self.params['l_intake'] = l_intake
        
        self.params['r_in_spiral'] = r_in_spiral
        self.params['t_out_spiral'] = t_out_spiral
        self.params['N_spirals_t'] = N_spirals_t
        
        
    def get_profile(self) -> Dict[np.ndarray, np.ndarray]:
        '''
        Get nacelle profile segments.
        
        Returns
        -------
        profile_x, profile_y : np.ndarray 
            Nacelle profile coordinates.
        '''
        self.calculate_segment_points()
        self.calculate_profile_segments()
        self.calculate_profile_features()
        
        # Define gear box, motor, and inverter outline
        self.profile_segments[32] = np.array([
            [self.segment_points[2][0], self.segment_points[0][1]],
            [self.segment_points[2][0], self.segment_points[2][1] - 0.5 * self.f_scale],
            [self.segment_points[2][0] + 8 * self.f_scale, self.segment_points[2][1] - 0.5 * self.f_scale],
            [self.segment_points[2][0] + 8 * self.f_scale, self.segment_points[0][1]]
        ])
        self.profile_segments[33] = np.array([
            [self.segment_points[2][0] + 10 * self.f_scale, self.segment_points[0][1]],
            [self.segment_points[2][0] + 10 * self.f_scale, self.segment_points[2][1] - 0.5 * self.f_scale],
            [self.segment_points[2][0] + 18 * self.f_scale, self.segment_points[2][1] - 0.5 * self.f_scale],
            [self.segment_points[2][0] + 18 * self.f_scale, self.segment_points[0][1]]
        ])
        self.profile_segments[34] = np.array([
            [self.segment_points[2][0] + 20 * self.f_scale, self.segment_points[0][1]],
            [self.segment_points[2][0] + 20 * self.f_scale, self.segment_points[2][1] - 0.5 * self.f_scale],
            [self.segment_points[2][0] + 28 * self.f_scale, self.segment_points[2][1] - 0.5 * self.f_scale],
            [self.segment_points[2][0] + 28 * self.f_scale, self.segment_points[0][1]]
        ])

        # Do not concatenate segments but rather plot separately to avoid ordering issues
        # profile_x = [self.profile_segments[0][:, 0]] + [self.profile_segments[i+1][1:, 0] for i in range(self.n_segment-1)]
        # profile_y = [self.profile_segments[0][:, 1]] + [self.profile_segments[i+1][1:, 1] for i in range(self.n_segment-1)]
        # self.profile_x = np.concatenate(profile_x)
        # self.profile_y = np.concatenate(profile_y)
        self.profile_x = [value[:, 0] for key, value in self.profile_segments.items()]
        self.profile_y = [value[:, 1] for key, value in self.profile_segments.items()]
        
        return self.profile_x, self.profile_y
    
    
    @property
    def n_point_profile(self) -> int:
        '''
        Number of points for the nacelle profile.
        '''
        return self.profile_x.shape[0]
    
    
    @property
    def n_segment(self) -> int:
        '''
        Number of nacelle profile segments.
        '''
        return len(self.profile_segments)
    
    
    def calculate_segment_points(self) -> None:
        '''
        Calculate nacelle profile segment end points based on the profile parameters.
        '''
        
        # Tweaking parameters for visual pleasure that are not a direct output of the EDF model
        # dx_te_rotor_to_le_inner_cowl = 5
        # dx_le_inner_cowl_to_le_outer_stator = 2
        # dx_te_outer_stator_to_in_hx = 3
        # dx_out_root_hx_to_in_inner_cowl_nozzle = 1
        
        # dy_le_tip_inner_stator_to_le_inner_cowl = 0.5
        # dy_le_inner_cowl_to_le_root_inner_stator = 0.5
        # dy_le_tip_rotor_to_le_highlight = 1  # optionally replaced with something like theta_droop/theta_scarf
        # dy_out_tip_hx_to_in_outer_cowl_nozzle = 0.5
        # dy_te_tip_inner_stator_to_in_inner_cowl_nozzle = 0.5
        
        self.f_scale = 0.05
        # self.f_scale = 1
        dx_te_rotor_to_le_inner_cowl = 7 * self.f_scale
        dx_le_inner_cowl_to_le_outer_stator = 3 * self.f_scale
        dx_te_outer_stator_to_in_hx = 3 * self.f_scale
        dx_out_root_hx_to_in_inner_cowl_nozzle = 1 * self.f_scale
        dx_out_hx_to_in_bp_nozzle = 5 * self.f_scale
        
        dy_le_tip_inner_stator_to_le_inner_cowl = 0.5 * self.f_scale
        dy_le_inner_cowl_to_le_root_inner_stator = 0.5 * self.f_scale
        dy_le_tip_rotor_to_le_highlight = -1 * self.f_scale  # optionally replaced with something like theta_droop/theta_scarf
        # dy_out_tip_hx_to_in_outer_cowl_nozzle = 0.5 * self.f_scale
        dy_out_tip_hx_to_in_outer_cowl_nozzle = 1 * self.f_scale
        dy_te_tip_inner_stator_to_in_inner_cowl_nozzle = 0.5 * self.f_scale
        
        # x-coordinates
        x1 = -self.params['l_spinner']
        x2 = 0.0
        x3 = self.params['l_rotor']
        x4 = x3 + dx_te_rotor_to_le_inner_cowl + dx_le_inner_cowl_to_le_outer_stator + self.params['l_stator'] + dx_te_outer_stator_to_in_hx + self.params['l_hx_side'] / 2 - self.params['l_stator'] / 2
        x5 = x4
        x6 = x3 + dx_te_rotor_to_le_inner_cowl
        x7 = x6 + dx_le_inner_cowl_to_le_outer_stator
        x8 = x7
        x9 = -self.params['l_intake']
        x10 = x4 + self.params['l_stator'] / 2 + self.params['l_hx_avg'] + self.params['l_hx_side'] / 2  #self.params['l_hx'] / 2
        x11 = x10 + dx_out_hx_to_in_bp_nozzle + self.params['l_bp_nozzle']
        x12 = x10
        x13 = x12 - self.params['l_hx_side']
        x14 = x13 - dx_te_outer_stator_to_in_hx - self.params['l_hx_avg']
        x15 = x14
        x16 = x15 + dx_te_outer_stator_to_in_hx  # x13
        x17 = x16 + self.params['l_hx_side']  # x12
        x18 = x17 + dx_out_root_hx_to_in_inner_cowl_nozzle
        # x19 = x18 + self.params['l_core_nozzle']
        x19 = x11
        x20 = x5 + self.params['l_stator']
        x21 = x20
        x22 = max(x11, x19)
        
        # Define body of revolution housing gear box, motor, and inverter
        self.zeta_max_core_body = self.params['r_hub_rotor'] / (x22 - x1)
        self.l_core_body = x22 - x1
        # =============================================================================
        last_ok, first_bad = find_beta_threshold(
            R_le_over_c=self.zeta_max_core_body / 2,
            Psi_zeta_max=0.35,
            zeta_max=self.zeta_max_core_body,
            zeta_te=0,
            step_deg=0.01,
        )
        print('last_ok =', last_ok)
        # =============================================================================
        # Vol, l, _, _, surface_area, Psi_list_closed, zeta_list_closed, _ = \
        out_dict = generate_streamlined_body_geometry(
            R_le_over_c=self.zeta_max_core_body / 2,
            beta_tail=last_ok,  # 30,
            Psi_zeta_max=0.35,
            zeta_max=self.zeta_max_core_body,
            zeta_te=0,
            dimensional_known_dict={'length': self.l_core_body}
        )
        Psi_list_closed = out_dict['Psi']
        zeta_list_closed = out_dict['zeta']
        
        print('Psi_list_closed, zeta_list_closed =', Psi_list_closed, zeta_list_closed)
        # xs, ys = Psi_list_closed[:int(len(Psi_list_closed) / 2)] * self.l_core_body, zeta_list_closed[:int(len(Psi_list_closed) / 2)] * self.l_core_body
        xs, ys = Psi_list_closed * self.l_core_body, zeta_list_closed * self.l_core_body
        xs -= self.params['l_spinner']
        
        # y-coordinates
        
        y1 = 0.0
        
        # Interpolate from body of revolution to ensure visual coherence
        def interpolate_core_body(x,y,x0,method='cubic'):
            """Return N points (xs,ys) equally spaced along arc-length between x0 and x1.
            Assumes x is sorted (monotonic). method='linear' or 'cubic' determines interpolation.
            """
            # cumulative arc-length parameter s
            s = np.hstack([0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
            # map x->s by linear interp (works if x is monotonic)
            s0 = np.interp(x0, x, s)
            # build interpolators y(s)
            if method=='linear':
                fy = interp1d(s, y, kind='linear', assume_sorted=True)
            else:
                fy = CubicSpline(s, y, bc_type='natural')
            y0 = fy(s0)
            return y0
        y2 = interpolate_core_body(xs,ys,x2,method='linear')
        y3 = interpolate_core_body(xs,ys,x3,method='linear')
        y4 = interpolate_core_body(xs,ys,x4,method='linear')
        y5 = y4 + self.params['h_stator_in']
        y6 = y5 + dy_le_tip_inner_stator_to_le_inner_cowl
        
        # Interpolate sections of inner cowl suction surface from
        # outer_cowl_profile() extending between 6 and 19 (not
        # super accurate but visually coherent geometry)
        #* (6,19)
        # y19 = self.params['r_core_nozzle_out']
        y19 = y6 - 0.1
        self.profile_segments[5] = self.outer_cowl_profile(np.array([x6, y6]), np.array([x19, y19]))  ##### TO-DO #####
        #* (6,7)
        self.profile_segments[23] = self.interp_cowl_profile(self.profile_segments[5], x6, x7)
        #* (7,15)
        self.profile_segments[24] = self.interp_cowl_profile(self.profile_segments[5], x7, x15)
        y7 = self.profile_segments[24][0,1]
        # y7 = y6 + dy_le_inner_cowl_to_le_root_inner_stator
        #* (15,16)
        self.profile_segments[25] = self.interp_cowl_profile(self.profile_segments[5], x15, x16)
        y15 = self.profile_segments[25][0,1]
        # y15 = y14 - self.params['h_stator_out']
        #* (16,17)
        self.profile_segments[26] = self.interp_cowl_profile(self.profile_segments[5], x16, x17)
        y16 = self.profile_segments[26][0,1]
        # y16 = y13 - h_hx
        #* (17,18)
        self.profile_segments[27] = self.interp_cowl_profile(self.profile_segments[5], x17, x18)
        y17 = self.profile_segments[27][0,1]
        # y17 = y16
        #* (18,19)
        self.profile_segments[28] = self.interp_cowl_profile(self.profile_segments[5], x18, x19)
        y18 = self.profile_segments[28][0,1]
        # y18 = y5 + dy_te_tip_inner_stator_to_in_inner_cowl_nozzle

        y8 = y7 + self.params['h_stator_out']
        y9 = y8 + dy_le_tip_rotor_to_le_highlight
        y10 = self.params['r_hx_out'] #+ dy_out_tip_hx_to_in_outer_cowl_nozzle
        # y11 = self.params['r_bp_nozzle_out']
        y11 = y10# - 0.1
        # y12 = self.params['r_hx_out']
        y12 = y17 + self.params['h_hx']
        # y12 = y8
        # y13 = y12
        y13 = y16 + self.params['h_hx']
        # y14 = y8
        y14 = y15 + self.params['h_stator_out']
        y20 = y5
        # y21 = y4
        y21 = interpolate_core_body(xs,ys,x21,method='linear')
        y22 = 0.0
        
        # Segment end points
        self.segment_points[0] = np.array([x1, y1])
        self.segment_points[1] = np.array([x2, y2])
        self.segment_points[2] = np.array([x3, y3])
        self.segment_points[3] = np.array([x4, y4])
        self.segment_points[4] = np.array([x5, y5])
        self.segment_points[5] = np.array([x6, y6])
        self.segment_points[6] = np.array([x7, y7])
        self.segment_points[7] = np.array([x8, y8])
        self.segment_points[8] = np.array([x9, y9])
        self.segment_points[9] = np.array([x10, y10])
        self.segment_points[10] = np.array([x11, y11])
        self.segment_points[11] = np.array([x12, y12])
        self.segment_points[12] = np.array([x13, y13])
        self.segment_points[13] = np.array([x14, y14])
        self.segment_points[14] = np.array([x15, y15])
        self.segment_points[15] = np.array([x16, y16])
        self.segment_points[16] = np.array([x17, y17])
        self.segment_points[17] = np.array([x18, y18])
        self.segment_points[18] = np.array([x19, y19])
        self.segment_points[19] = np.array([x20, y20])
        self.segment_points[20] = np.array([x21, y21])
        self.segment_points[21] = np.array([x22, y22])
        
        
    def calculate_profile_segments(self) -> None:
        '''
        Calculate nacelle profile segments:
        
        - Spinner (conical surface): (0,1);
        - Fan face intake (planar surface): (1,2);
        - Inlet surface: (3,2);
        - Outer cowl surface: (3,4);
        - Bypass duct outer surface: (5,4);
        - Bypass duct inlet (planar surface): (5,6);
        - Bypass duct inner surface: (6,7);
        - Core cowl (conical surface): (7, 8);
        - Core duct outer surface: (9,8);
        - Core duct inlet (planar surface): (9,10);
        - Core duct inner surface: (10,11);
        - Core plug (conical surface): (11,12);
        '''
        
        #* (1,2)
        self.profile_segments[0] = self.core_body_profile(
            self.segment_points[0][0], self.segment_points[0][1],
            self.segment_points[1][0], self.segment_points[1][1]
        )
        
        #* (2,3)
        self.profile_segments[1] = self.core_body_profile(
            self.segment_points[1][0], self.segment_points[1][1],
            self.segment_points[2][0], self.segment_points[2][1]
        )
        
        #* (3,4)
        self.profile_segments[2] = self.core_body_profile(
            self.segment_points[2][0], self.segment_points[2][1],
            self.segment_points[3][0], self.segment_points[3][1]
        )
        
        #* (4,21)
        self.profile_segments[29] = self.core_body_profile(
            self.segment_points[3][0], self.segment_points[3][1],
            self.segment_points[20][0], self.segment_points[20][1]
        )
        
        #* (5,6) ##### CHANGED ORDER TO RUN self.outer_cowl_profile() FIRST
        # self.profile_segments[4] = self.inlet_profile(self.segment_points[5], self.segment_points[19], flip=False)  # False
        self.profile_segments[4] = self.inlet_profile(self.segment_points[5], self.segment_points[4], flip=False)  # False
        
        #* (4,5) inner stator LE profile
        t = np.linspace(0.0, 1.0, 50)
        def quadratic_bezier(p0, p1, p2, t):
            """Evaluate quadratic Bézier at parameter array t (shape (m,)).
            Returns array shape (m,2).
            """
            t = np.asarray(t)
            one_minus_t = 1.0 - t
            return (one_minus_t**2)[:,None]*p0 + (2*one_minus_t*t)[:,None]*p1 + (t**2)[:,None]*p2
        le_tip = self.segment_points[4]
        le_hub = self.segment_points[3]
        le_tip[0] -= 0.5 * self.f_scale
        le_hub[0] -= 0.5 * self.f_scale
        avg_le = np.average((self.segment_points[3], le_tip), axis=0)
        avg_le[0] += 0.5 * self.f_scale
        self.profile_segments[3] = quadratic_bezier(
            np.asarray(le_hub, dtype=float),
            np.asarray(avg_le, dtype=float),
            np.asarray(le_tip, dtype=float),
            t,
        )
        
        #* (7,8) outer stator LE profile
        le_tip = self.segment_points[7]
        le_hub = self.segment_points[6]
        le_tip[0] -= 0.5 * self.f_scale
        le_hub[0] -= 0.5 * self.f_scale
        avg_le = np.average((self.segment_points[6], le_tip), axis=0)
        avg_le[0] += 0.5 * self.f_scale
        self.profile_segments[6] = quadratic_bezier(
            np.asarray(le_hub, dtype=float),
            np.asarray(avg_le, dtype=float),
            np.asarray(le_tip, dtype=float),
            t,
        )
        
        #* (9,11) skip control point 10 for now (should be re-introduced to allow
        # realistic display of variable-area nozzle)
        self.profile_segments[8] = self.outer_cowl_profile(self.segment_points[8], self.segment_points[10])
        
        #* (8,9) ##### CHANGED ORDER TO RUN self.outer_cowl_profile() FIRST
        self.profile_segments[7] = self.inlet_profile(self.segment_points[8], self.segment_points[7], flip=False)#, target='outer')  # False
        
        #* (2,3) along rotor profile
        inlet_profile = np.flip(self.profile_segments[7], axis=0)
        rotor_tip_profile = self.interp_inlet_profile(
            inlet_profile, self.segment_points[1][0], self.segment_points[2][0]
        )
        # LE
        # =============================================================================
        # le_tip = rotor_tip_profile[0]  # for rotor tip to touch inner intake
        le_tip = [
            self.segment_points[1][0],
            self.params['r_tip_rotor'],
        ]  # for rotor tip to be of correct length
        # =============================================================================
        le_tip[0] += 1 * self.f_scale
        avg_le = np.average((self.segment_points[1], le_tip), axis=0)
        avg_le[0] -= 2 * self.f_scale
        avg_le[1] -= 1 * self.f_scale
        self.profile_segments[30] = quadratic_bezier(
            np.asarray(self.segment_points[1], dtype=float),
            np.asarray(avg_le, dtype=float),
            np.asarray(le_tip, dtype=float),
            t,
        )
        # TE
        # =============================================================================
        # te_tip = rotor_tip_profile[-1]  # for rotor tip to touch inner intake
        te_tip = [
            self.segment_points[2][0],
            self.params['r_tip_rotor'],
        ]  # for rotor tip to be of correct length
        # =============================================================================
        te_tip[0] -= 1 * self.f_scale
        avg_te = np.average((self.segment_points[2], te_tip), axis=0)
        avg_te[0] += 2 * self.f_scale
        avg_te[1] -= 1 * self.f_scale
        self.profile_segments[31] = quadratic_bezier(
            np.asarray(self.segment_points[2], dtype=float),
            np.asarray(avg_te, dtype=float),
            np.asarray(te_tip, dtype=float),
            t,
        )
        # =============================================================================
        self.profile_segments[35] = np.array([
            [le_tip[0], le_tip[1]],
            [te_tip[0], te_tip[1]],
        ])
        # =============================================================================
        
        #* (10,11) skip control point 10 for now (should be re-introduced to allow
        self.profile_segments[9] = np.array([[np.nan, np.nan]])
        
        #* (11,8) use spline with tangency constraints
        # self.profile_segments[10] = self.bypass_duct_outer_profile(self.segment_points[10], self.segment_points[8], self.segment_points[11])
        control_points = np.array([
            self.segment_points[11],
            self.segment_points[12],
            self.segment_points[13],
        ])
        control_points_ordered = np.array([
            self.segment_points[13],
            self.segment_points[12],
            self.segment_points[11],
        ])
        # Calculate angle beta between horizontal and suction surface of outer airfoil at TE
        '''     
                 SS
             α `
        PS  \   `  γ = β - α
              \  `
             γ  \ `  β
                  \`
            --------` TE
            HORZ
        ''' 
        self.beta_outer = np.arctan(np.diff(self.profile_segments[8][-2:,1]) / np.diff(self.profile_segments[8][-2:,0])) * 180 / np.pi
        self.profile_segments[10] = self.bypass_duct_inner_profile(control_points, self.segment_points[10], self.segment_points[7], self.beta_outer)
        # self.profile_segments[10] = self.bypass_duct_inner_profile(control_points_ordered, self.segment_points[7], self.segment_points[10], self.beta_outer)
        profile_segments_10_noflap = self.bypass_duct_inner_profile(control_points_ordered, self.segment_points[7], self.segment_points[10], self.beta_outer)
        
        # =============================================================================
        # Interpolate sections of outer cowl pressure surface from
        # bypass_duct_inner_profile() extending between 11 and 8 (not
        # super accurate but visually coherent geometry)
        
        #* (11,12)
        self.profile_segments[36] = self.interp_bypass_duct_inner_profile(
            # self.profile_segments[10], self.segment_points[10][0], self.segment_points[11][0]
            # self.profile_segments[10], self.segment_points[11][0], self.segment_points[10][0]
            profile_segments_10_noflap, self.segment_points[11][0], self.segment_points[10][0]
        )
        #* (12,13)
        self.profile_segments[37] = self.interp_bypass_duct_inner_profile(
            # self.profile_segments[10], self.segment_points[11][0], self.segment_points[12][0]
            # self.profile_segments[10], self.segment_points[12][0], self.segment_points[11][0]
            profile_segments_10_noflap, self.segment_points[12][0], self.segment_points[11][0]
        )
        #* (13,14)
        self.profile_segments[38] = self.interp_bypass_duct_inner_profile(
            # self.profile_segments[10], self.segment_points[12][0], self.segment_points[13][0]
            # self.profile_segments[10], self.segment_points[13][0], self.segment_points[12][0]
            profile_segments_10_noflap, self.segment_points[13][0], self.segment_points[12][0]
        )
        #* (14,8)
        self.profile_segments[39] = self.interp_bypass_duct_inner_profile(
            # self.profile_segments[10], self.segment_points[13][0], self.segment_points[7][0]
            # self.profile_segments[10], self.segment_points[7][0], self.segment_points[13][0]
            profile_segments_10_noflap, self.segment_points[7][0], self.segment_points[13][0]
        )
        # =============================================================================
        
        # =============================================================================
        angle_hinge = np.rad2deg(-self.params['beta_bp_nozzle'])
        x_hinge = self.segment_points[10][0] - self.params['l_bp_nozzle']
        _foil_xu = self.profile_segments[8][:, 0]
        _foil_xl = np.flip(self.profile_segments[10][:, 0])
        # # =============================================================================
        # _foil_xl = np.hstack((
        #     np.flip(self.profile_segments[7][:, 0]),
        #     np.flip(self.profile_segments[10][:, 0]),
        # ))
        # # =============================================================================
        _foil_yu = self.profile_segments[8][:, 1]
        _foil_yl = np.flip(self.profile_segments[10][:, 1])
        # # =============================================================================
        # _foil_yl = np.hstack((
        #     np.flip(self.profile_segments[7][:, 1]),
        #     np.flip(self.profile_segments[10][:, 1]),
        # ))
        # # =============================================================================
        # print('_foil_xu, _foil_xl, _foil_yu, _foil_yl =', _foil_xu, _foil_xl, _foil_yu, _foil_yl)
        print('len(_foil_xu), len(_foil_xl), len(_foil_yu), len(_foil_yl) =', len(_foil_xu), len(_foil_xl), len(_foil_yu), len(_foil_yl))
        self.xu_bp, self.xl_bp, self.yu_bp, self.yl_bp = self.add_flap_to_airfoil(
            _foil_xu, _foil_xl, _foil_yu, _foil_yl, angle_hinge, x_hinge,
        )
        
        self.xl_bp = np.hstack((np.flip(self.profile_segments[7][:, 0]), self.xl_bp))
        self.yl_bp = np.hstack((np.flip(self.profile_segments[7][:, 1]), self.yl_bp))
        # =============================================================================
        
        #* (12,17) HX TE
        self.profile_segments[12] = self.create_straight_line(
            self.segment_points[11][0], self.segment_points[11][1],
            self.segment_points[16][0], self.segment_points[16][1]
        )
        
        #* (13,16)) HX LE
        self.profile_segments[14] = self.create_straight_line(
            self.segment_points[12][0], self.segment_points[12][1],
            self.segment_points[15][0], self.segment_points[15][1]
        )
        
        #* (14,15) outer stator TE profile
        te_tip = self.segment_points[13]
        te_hub = self.segment_points[14]
        te_tip[0] += 0.5 * self.f_scale
        te_hub[0] += 0.5 * self.f_scale
        avg_te = np.average((self.segment_points[14], te_tip), axis=0)
        avg_te[0] -= 0.5 * self.f_scale
        self.profile_segments[15] = quadratic_bezier(
            np.asarray(te_hub, dtype=float),
            np.asarray(avg_te, dtype=float),
            np.asarray(te_tip, dtype=float),
            t,
        )
        
        #* (19,5) use spline with tangency constraints but without control points
        control_points = np.array([self.segment_points[19]])
        # Calculate angle beta between horizontal and suction surface of inner airfoil at TE
        self.beta_inner = np.arctan(np.diff(self.profile_segments[5][-2:,1]) / np.diff(self.profile_segments[5][-2:,0])) * 180 / np.pi
        # self.profile_segments[20] = self.bypass_duct_inner_profile(control_points, self.segment_points[18], self.segment_points[19], self.beta_inner)
        self.profile_segments[20] = self.bypass_duct_inner_profile(control_points, self.segment_points[18], self.segment_points[4], self.beta_inner)
        profile_segments_20_ordered = self.bypass_duct_inner_profile(control_points, self.segment_points[4], self.segment_points[18], self.beta_inner)
        
        #* (20,5)
        self.profile_segments[40] = self.interp_bypass_duct_inner_profile(
            # self.profile_segments[20], self.segment_points[19][0], self.segment_points[4][0]
            profile_segments_20_ordered, self.segment_points[4][0], self.segment_points[19][0]
        )
        
        # =============================================================================
        angle_hinge = np.rad2deg(-self.params['beta_core_nozzle'])
        x_hinge = self.segment_points[18][0] - self.params['l_core_nozzle']
        _foil_xu = self.profile_segments[5][:, 0]
        _foil_xl = np.flip(self.profile_segments[20][:, 0])
        # # =============================================================================
        # _foil_xl = np.hstack((
        #     np.flip(self.profile_segments[4][:, 0]),
        #     np.flip(self.profile_segments[20][:, 0]),
        # ))
        # # =============================================================================
        _foil_yu = self.profile_segments[5][:, 1]
        _foil_yl = np.flip(self.profile_segments[20][:, 1])
        # # =============================================================================
        # _foil_yl = np.hstack((
        #     np.flip(self.profile_segments[4][:, 1]),
        #     np.flip(self.profile_segments[20][:, 1]),
        # ))
        # # =============================================================================
        self.xu_core, self.xl_core, self.yu_core, self.yl_core = self.add_flap_to_airfoil(
            _foil_xu, _foil_xl, _foil_yu, _foil_yl, angle_hinge, x_hinge,
        )
        
        # self.xu_core = np.hstack((np.flip(self.profile_segments[4][:, 0]), self.xu_core))
        # self.yu_core = np.hstack((np.flip(self.profile_segments[4][:, 1]), self.yu_core))
        self.xl_core = np.hstack((np.flip(self.profile_segments[4][:, 0]), self.xl_core))
        self.yl_core = np.hstack((np.flip(self.profile_segments[4][:, 1]), self.yl_core))
        # =============================================================================
        
        #* (20,21) inner stator TE profile
        te_tip = self.segment_points[19]
        te_hub = self.segment_points[20]
        te_tip[0] += 0.5 * self.f_scale
        te_hub[0] += 0.5 * self.f_scale
        avg_te = np.average((self.segment_points[20], te_tip), axis=0)
        avg_te[0] -= 0.5 * self.f_scale
        self.profile_segments[21] = quadratic_bezier(
            np.asarray(te_hub, dtype=float),
            np.asarray(avg_te, dtype=float),
            np.asarray(te_tip, dtype=float),
            t,
        )
        
        #* (21,22)
        self.profile_segments[22] = self.core_body_profile(
            self.segment_points[20][0], self.segment_points[20][1],
            self.segment_points[21][0], self.segment_points[21][1]
        )


    def calculate_profile_features(self) -> None:
        '''
        Calculate nacelle profile features:
        
        - l_fore: fore-body length;
        - l_lip: lip length;
        
        - r_if: initial fore-body radius;
        - r_throat: inlet throat radius;
        - r_max: maximum radius;

        - theta_airfoil: airfoil geometric incidence angle (degree);
        - theta_bp = bypass exit angle (degree);
        - theta_cr = core exit angle (degree);
        - theta_nac = nacelle boat tail angle (degree);
        
        - x_max: Axial location of maximum radius;
        - x_thr: Axial location of inlet throat;
        '''
        pass


    def create_straight_line(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        '''
        Generate a straight line
        
        Parameters
        ----------
        x0, y0 : float
            Starting point coordinates.
            
        x1, y1 : float
            Ending point coordinates.
            
        Returns
        -------
        line : np.ndarray [n_point_segment, 2]
            Straight line coordinates.
        '''
        line = np.zeros((self.n_point_segment, 2))
        line[:, 0] = np.linspace(x0, x1, self.n_point_segment, endpoint=True)
        
        if abs(x0 - x1) < 1e-6:
            line[:, 1] = np.linspace(y0, y1, self.n_point_segment, endpoint=True)
        else:
            line[:, 1] = y0 + (y1 - y0) / (x1 - x0) * (line[:, 0] - x0)
        
        return line
    
    
    def core_body_profile(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        """Define core body profile and split into 5 segments with self.n_point_segment points each."""
        
        # =============================================================================
        last_ok, first_bad = find_beta_threshold(
            R_le_over_c=self.zeta_max_core_body / 2,
            Psi_zeta_max=0.35,
            zeta_max=self.zeta_max_core_body,
            zeta_te=0,
            step_deg=0.01,
        )
        # =============================================================================
        # Vol, l, _, _, surface_area, Psi_list_closed, zeta_list_closed, _ = \
        out_dict = generate_streamlined_body_geometry(
            R_le_over_c=self.zeta_max_core_body / 2,
            beta_tail=last_ok,  # 30,
            Psi_zeta_max=0.35,
            zeta_max=self.zeta_max_core_body,
            zeta_te=0,
            dimensional_known_dict={'length': self.l_core_body},
        )
        Psi_list_closed = out_dict['Psi']
        zeta_list_closed = out_dict['zeta']
            
        # xs, ys = Psi_list_closed[:int(len(Psi_list_closed) / 2)] * self.l_core_body, zeta_list_closed[:int(len(Psi_list_closed) / 2)] * self.l_core_body
        xs, ys = Psi_list_closed * self.l_core_body, zeta_list_closed * self.l_core_body
        xs -= self.params['l_spinner']
        
        def resample_between_x(x,y,x0,x1,N,method='cubic'):
            """Return N points (xs,ys) equally spaced along arc-length between x0 and x1.
            Assumes x is sorted (monotonic). method='linear' or 'cubic' determines interpolation.
            """
            # cumulative arc-length parameter s
            s = np.hstack([0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
            # ensure increasing x
            if x1 < x0:
                x0,x1 = x1,x0
            # map x->s by linear interp (works if x is monotonic)
            s0, s1 = np.interp([x0, x1], x, s)
            ss = np.linspace(s0, s1, N)
            # build interpolators x(s), y(s)
            if method=='linear':
                fx = interp1d(s, x, kind='linear', assume_sorted=True)
                fy = interp1d(s, y, kind='linear', assume_sorted=True)
            else:
                fx = CubicSpline(s, x, bc_type='natural')
                fy = CubicSpline(s, y, bc_type='natural')
            xs = fx(ss); ys = fy(ss)
            return xs, ys
        
        xs_resampled, ys_resampled = resample_between_x(xs,ys,x0,x1,self.n_point_segment,method='linear')
        line = np.zeros((self.n_point_segment, 2))
        line[:, 0] = xs_resampled
        line[:, 1] = ys_resampled
        
        return line
    

    def interp_cowl_profile(self, cowl_profile: np.ndarray, x0: float, x1: float) -> np.ndarray:
        """Interpolate cowl profile at self.n_point_segment points between x0 and x1."""
        
        def resample_between_x(x,y,x0,x1,N,method='cubic'):
            """Return N points (xs,ys) equally spaced along arc-length between x0 and x1.
            Assumes x is sorted (monotonic). method='linear' or 'cubic' determines interpolation.
            """
            # cumulative arc-length parameter s
            s = np.hstack([0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
            # ensure increasing x
            if x1 < x0:
                x0,x1 = x1,x0
            # map x->s by linear interp (works if x is monotonic)
            s0, s1 = np.interp([x0, x1], x, s)
            ss = np.linspace(s0, s1, N)
            # build interpolators x(s), y(s)
            if method=='linear':
                fx = interp1d(s, x, kind='linear', assume_sorted=True)
                fy = interp1d(s, y, kind='linear', assume_sorted=True)
            else:
                fx = CubicSpline(s, x, bc_type='natural')
                fy = CubicSpline(s, y, bc_type='natural')
            xs = fx(ss); ys = fy(ss)
            return xs, ys
        
        xs_resampled, ys_resampled = resample_between_x(cowl_profile[:, 0],cowl_profile[:, 1],x0,x1,self.n_point_segment,method='linear')
        line = np.zeros((self.n_point_segment, 2))
        line[:, 0] = xs_resampled
        line[:, 1] = ys_resampled
        
        return line
    
    
    def interp_inlet_profile(self, inlet_profile: np.ndarray, x0: float, x1: float) -> np.ndarray:
        """Interpolate cowl inlet profile at self.n_point_segment points between x0 and x1."""
        
        def resample_between_x(x,y,x0,x1,N,method='cubic'):
            """Return N points (xs,ys) equally spaced along arc-length between x0 and x1.
            Assumes x is sorted (monotonic). method='linear' or 'cubic' determines interpolation.
            """
            # cumulative arc-length parameter s
            s = np.hstack([0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
            # ensure increasing x
            if x1 < x0:
                x0,x1 = x1,x0
            # map x->s by linear interp (works if x is monotonic)
            s0, s1 = np.interp([x0, x1], x, s)
            ss = np.linspace(s0, s1, N)
            # build interpolators x(s), y(s)
            if method=='linear':
                fx = interp1d(s, x, kind='linear', assume_sorted=True)
                fy = interp1d(s, y, kind='linear', assume_sorted=True)
            else:
                fx = CubicSpline(s, x, bc_type='natural')
                fy = CubicSpline(s, y, bc_type='natural')
            xs = fx(ss); ys = fy(ss)
            return xs, ys
        
        xs_resampled, ys_resampled = resample_between_x(inlet_profile[:, 0],inlet_profile[:, 1],x0,x1,self.n_point_segment,method='linear')
        line = np.zeros((self.n_point_segment, 2))
        line[:, 0] = xs_resampled
        line[:, 1] = ys_resampled
        
        return line
    
    
    def interp_bypass_duct_inner_profile(self, bypass_duct_inner_profile: np.ndarray, x0: float, x1: float) -> np.ndarray:
        
        def resample_between_x(x,y,x0,x1,N,method='cubic'):
            """Return N points (xs,ys) equally spaced along arc-length between x0 and x1.
            Assumes x is sorted (monotonic). method='linear' or 'cubic' determines interpolation.
            """
            # cumulative arc-length parameter s
            s = np.hstack([0.0, np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
            # ensure increasing x
            if x1 < x0:
                x0,x1 = x1,x0
            # map x->s by linear interp (works if x is monotonic)
            s0, s1 = np.interp([x0, x1], x, s)
            ss = np.linspace(s0, s1, N)
            # build interpolators x(s), y(s)
            if method=='linear':
                fx = interp1d(s, x, kind='linear', assume_sorted=True)
                fy = interp1d(s, y, kind='linear', assume_sorted=True)
            else:
                fx = CubicSpline(s, x, bc_type='natural')
                fy = CubicSpline(s, y, bc_type='natural')
            xs = fx(ss); ys = fy(ss)
            return xs, ys
        
        xs_resampled, ys_resampled = resample_between_x(
            bypass_duct_inner_profile[:, 0],bypass_duct_inner_profile[:, 1],x0,x1,self.n_point_segment,method='linear'
        )
        line = np.zeros((self.n_point_segment, 2))
        line[:, 0] = xs_resampled
        line[:, 1] = ys_resampled
        
        return line
    
    
    def add_flap_to_airfoil(self,
        _xu, _xl, _yu, _yl, angle_hinge, x_hinge,
    ) -> tuple:
        """
        Add a flap (variable-area nozzle in this case)
        at x=x_hinge to upper and lower surface. Based on
        section_flap() in cst_modeling/tools/auxiliary.py.
        """
        
        xu_, xl_, yu_, yl_ = transform(_xu, _xl, _yu, _yl, rot=angle_hinge, x0=x_hinge)
        
        iu1, iu2, _ = intersect_index(_xu, _yu, xu_, yu_)
        il1, il2, _ = intersect_index(_xl, _yl, xl_, yl_)
        nn = len(_xu)
        nu_flap = nn - iu1
        nl_flap = nn - il1
    
        #* Adjust number of points on the flap
        xu_new2 = np.concatenate((_xu[:iu1], xu_[iu2:]), axis=0)
        yu_new2 = np.concatenate((_yu[:iu1], yu_[iu2:]), axis=0)
        xl_new2 = np.concatenate((_xl[:il1], xl_[il2:]), axis=0)
        yl_new2 = np.concatenate((_yl[:il1], yl_[il2:]), axis=0)
    
        xx_u = np.linspace(_xu[iu1], xu_[-1], nu_flap)
        yy_u = interp_from_curve(xx_u, xu_new2, yu_new2)
        xu_new = np.concatenate((_xu[:iu1], xx_u), axis=0)
        yu_new = np.concatenate((_yu[:iu1], yy_u), axis=0)
    
        xx_l = np.linspace(_xl[il1], xl_[-1], nl_flap)
        yy_l = interp_from_curve(xx_l, xl_new2, yl_new2, extrapolate=True)
        xl_new = np.concatenate((_xl[:il1], xx_l), axis=0)
        yl_new = np.concatenate((_yl[:il1], yy_l), axis=0)
        
        return (xu_new, xl_new, yu_new, yl_new)
    

    def outer_cowl_profile(self, highlight_point, te_point) -> None:
        '''
        Generate outer cowl curve, i.e., profile segment (3) between segment points (3,4).
        '''
        
        #* Airfoil profile
        self._xx, self._yu, self._yl, _, _ = cst_foil(
                self.n_point_segment, self.params['cst_u'], self.params['cst_l'])
        
        self.SS_TE_angle = np.arctan(np.diff(self._yu[-2:])/np.diff(self._xx[-2:])) * 180 / np.pi
        self.PS_TE_angle = np.arctan(np.diff(self._yl[-2:])/np.diff(self._xx[-2:])) * 180 / np.pi
        self.alpha = self.SS_TE_angle - self.PS_TE_angle  # see sketch of airfoil TE in docstring above
        
        scale = np.linalg.norm(te_point - highlight_point)
        # =============================================================================
        angle = np.arctan2( te_point[1] - highlight_point[1],
                            te_point[0] - highlight_point[0])
        angle = np.rad2deg(angle)
        # angle = 0.0
        # =============================================================================
        
        self._foil_xu, self._foil_xl, self._foil_yu, self._foil_yl = transform(
                    self._xx, self._xx, self._yu, self._yl, scale=scale, rot=angle, 
                    dx=highlight_point[0], dy=highlight_point[1])
        
        self.features['theta_airfoil'] = angle
        self.features['scale_airfoil'] = scale
        
        return np.concatenate(
            (self._foil_xu[:, None], self._foil_yu[:, None]), axis=1
        )


    def inlet_profile(self, highlight_point, fan_face_point, flip: bool = False) -> None:
        '''
        Generate inlet curve, i.e., profile segment (2) between segment points (2,3).
        
        Must run after `outer_cowl_profile`.
        '''
        
        xx = dist_clustcos(self.n_point_segment, a0=0.5, a1=0.99, beta=1.0)
        
        curve = np.zeros((self.n_point_segment, 2))
        curve[:, 0] = xx * (highlight_point[0] - fan_face_point[0]) + fan_face_point[0]
        
        scale = 1.0
        
        # sign controls vertical flip; +1 = original, -1 = flipped
        sign = -1.0 if flip else 1.0
        
        #* Solve scale to match the fan radius
        
        if fan_face_point[1] is not None:

            def func(scale):

                _, _foil_xl, _, _foil_yl = transform(
                            self._xx, self._xx, scale * self._yl, scale * self._yl, 
                            scale=self.features['scale_airfoil'], 
                            rot=self.features['theta_airfoil'], 
                            dx=highlight_point[0], dy=highlight_point[1])
                
                y = interp_from_curve(fan_face_point[0], _foil_xl, _foil_yl, extrapolate=True)
                
                res = abs(y-fan_face_point[1])
                # print('res =', res)
                return res
                
            root = fsolve(func, x0=1.0)
            scale = root[0]
            
            # print('> Inlet surface scale: %.2f'%(scale))
        
        _, _foil_xl, _, _foil_yl = transform(
                    self._xx, self._xx, sign * scale * self._yl, sign * scale * self._yl,
                    scale=self.features['scale_airfoil'], 
                    rot=self.features['theta_airfoil'], 
                    dx=highlight_point[0], dy=highlight_point[1])
        
        curve[:, 1] = interp_from_curve(curve[:, 0], _foil_xl, _foil_yl, extrapolate=True)
        
        return curve


    def bypass_duct_outer_profile(self, te_point, le_point, fan_exit_point) -> None:
        '''
        Generate bypass duct outer surface curve, i.e., profile segment (4) between segment points (4,5).
        
        Must run after `outer_cowl_profile`.
        '''
        xx = dist_clustcos(self.n_point_segment, a0=0.1, a1=0.9, beta=1.0)
        
        curve = np.zeros((self.n_point_segment, 2))
        curve[:, 0] = xx * (fan_exit_point[0] - te_point[0]) + te_point[0]
        
        scale = 1.0
        
        #* Solve scale to match the fan radius
        
        if fan_exit_point[1] is not None:

            def func(scale):
                
                _, _foil_xl, _, _foil_yl = transform(
                            self._xx, self._xx, scale*self._yl, scale*self._yl, 
                            scale=self.features['scale_airfoil'], 
                            rot=self.features['theta_airfoil'], 
                            dx=le_point[0], dy=le_point[1])
                
                y = interp_from_curve(fan_exit_point[0], _foil_xl, _foil_yl, extrapolate=True)
                
                return abs(y-fan_exit_point[1])
                
            root = fsolve(func, x0=1.0)
            scale = root[0]
            
            # print('> Bypass duct outer surface scale: %.2f'%(scale))
        
        _, _foil_xl, _, _foil_yl = transform(
                    self._xx, self._xx, scale*self._yl, scale*self._yl, 
                    scale=self.features['scale_airfoil'], 
                    rot=self.features['theta_airfoil'], 
                    dx=le_point[0], dy=le_point[1])
        
        curve[:, 1] = interp_from_curve(curve[:, 0], _foil_xl, _foil_yl, extrapolate=True)
        
        return curve


    def bypass_duct_inner_profile(self, control_points: np.ndarray, start, end, beta) -> None:
        '''
        Generate bypass duct inner surface curve, i.e., profile segment (6) between segment points (6,7).
        '''       
        # Angle between horizontal and airfoil PS
        self.gamma = beta - self.alpha
        bc_type = ((1, 0), (1, self.gamma[0] * np.pi / 180))

        xs = np.array([start[0]] + [x for x, _ in control_points] + [end[0]])
        ys = np.array([start[1]] + [y for _, y in control_points] + [end[1]])
        xs_sorted = xs[np.argsort(xs)]
        ys_sorted = ys[np.argsort(xs)]
        
        func = CubicSpline(xs_sorted, ys_sorted, bc_type=bc_type)

        curve = np.zeros((self.n_point_segment, 2))
        curve[:, 0] = np.linspace(start[0], end[0], self.n_point_segment, endpoint=True)
        curve[:, 1] = func(curve[:, 0])
        
        return curve


    def core_duct_outer_profile(self) -> None:
        '''
        Generate core duct outer surface curve, i.e., profile segment (8) between segment points (8,9).
        '''       
        
        bcx0 = (1, 0.0)
        bcx1 = (1, (self.segment_points[12][1] - self.segment_points[11][1]) / (self.segment_points[12][0] - self.segment_points[11][0]))
        
        bc_type = (bcx0, bcx1)
        
        control_points = copy.deepcopy(self.params['core_outer_control_points'])
        
        xs = [self.segment_points[9][0]] + [x for x, _ in control_points] + [self.segment_points[8][0]]
        ys = [self.segment_points[9][1]] + [y for _, y in control_points] + [self.segment_points[8][1]]
        
        func = CubicSpline(xs, ys, bc_type=bc_type)

        curve = np.zeros((self.n_point_segment, 2))
        curve[:, 0] = np.linspace(self.segment_points[8][0], self.segment_points[9][0], self.n_point_segment, endpoint=True)
        curve[:, 1] = func(curve[:, 0])
        
        self.profile_segments[8] = curve


    def core_duct_inner_profile(self) -> None:
        '''
        Generate core duct inner surface curve, i.e., profile segment (10) between segment points (10,11).
        '''       
        
        bcx0 = (1, 0.0)
        bcx1 = (1, (self.segment_points[12][1] - self.segment_points[11][1]) / (self.segment_points[12][0] - self.segment_points[11][0]))
        
        bc_type = (bcx0, bcx1)
        
        control_points = copy.deepcopy(self.params['core_outer_control_points'])
        
        xs = [self.segment_points[10][0]] + [x for x, _ in control_points] + [self.segment_points[11][0]]
        ys = [self.segment_points[10][1]] + [y for _, y in control_points] + [self.segment_points[11][1]]
        
        func = CubicSpline(xs, ys, bc_type=bc_type)

        curve = np.zeros((self.n_point_segment, 2))
        curve[:, 0] = np.linspace(self.segment_points[10][0], self.segment_points[11][0], self.n_point_segment, endpoint=True)
        curve[:, 1] = func(curve[:, 0])
        
        self.profile_segments[10] = curve


    def plot_side_view(self, fig=None, ax=None, show=True, color='k', alpha=0.25) -> None:
        '''
        Plot side view.
        '''
        
        if (fig == None or ax == None):
            fig, ax = plt.subplots(figsize=(16, 8))
        
        for i in range(len(self.profile_x)):
            ax.plot(self.profile_x[i], self.profile_y[i], color=color, lw=1, alpha=0.5)
        
        mirrored_profile_y = [seg * -1 for seg in self.profile_y]
        for i in range(len(self.profile_x)):
            ax.plot(self.profile_x[i], mirrored_profile_y[i], color=color, lw=1, alpha=0.5)
            
        # # =============================================================================
        # ax.plot(self.xu_bp, self.yu_bp, color=color)
        # ax.plot(self.xl_bp, self.yl_bp, color=color)
        
        # mirrored_profile_y = [seg * -1 for seg in [self.yu_bp, self.yl_bp]]
        # ax.plot(self.xu_bp, mirrored_profile_y[0], color=color)
        # ax.plot(self.xl_bp, mirrored_profile_y[1], color=color)
        
        # ##
        
        # ax.plot(self.xu_core, self.yu_core, color=color)
        # ax.plot(self.xl_core, self.yl_core, color=color)
        
        # mirrored_profile_y = [seg * -1 for seg in [self.yu_core, self.yl_core]]
        # ax.plot(self.xu_core, mirrored_profile_y[0], color=color)
        # ax.plot(self.xl_core, mirrored_profile_y[1], color=color)
        # # =============================================================================
        
        # ax.scatter(self.segment_points[10][0], self.params['r_bp_nozzle_out'], color=color, marker='.')
        # ax.scatter(self.segment_points[18][0], self.params['r_core_nozzle_out'], color=color, marker='.')
        
        ax.set_aspect('equal')
        
        # def add_label(x, y, label, dx=0.01, dy=0.01, color='b'):
        #     ax.plot(x, y, color+'.')
        #     ax.text(x+dx, y+dy, label, color=color)
        
        # for i_point in range(len(self.segment_points)):
            
        #     add_label(self.segment_points[i_point][0], self.segment_points[i_point][1], str(i_point))
        
        ##
        
        centre_body = np.vstack([
            self.profile_segments[0],
            self.profile_segments[1],
            self.profile_segments[2],
            self.profile_segments[29],
            self.profile_segments[22],
        ])
        rotor = np.vstack([
            self.profile_segments[30],
            self.profile_segments[35],
            np.flip(self.profile_segments[31], axis=0),
            np.flip(self.profile_segments[1], axis=0),
        ])
        inner_cowl = np.vstack([
            self.profile_segments[4],
            self.profile_segments[23],
            self.profile_segments[24],
            self.profile_segments[25],
            self.profile_segments[26],
            self.profile_segments[27],
            self.profile_segments[28],
            self.profile_segments[20],
        ])
        outer_cowl = np.vstack([
            self.profile_segments[7],
            self.profile_segments[8],
            self.profile_segments[10],
        ])
        outer_stator = np.vstack([
            self.profile_segments[6],
            np.flip(self.profile_segments[39], axis=0),
            np.flip(self.profile_segments[15], axis=0),
            np.flip(self.profile_segments[24], axis=0),
        ])
        hx = np.vstack([
            np.flip(self.profile_segments[14], axis=0),
            self.profile_segments[37],
            self.profile_segments[12],
            np.flip(self.profile_segments[26], axis=0),
        ])
        inner_stator = np.vstack([
            self.profile_segments[3],
            np.flip(self.profile_segments[40], axis=0),
            np.flip(self.profile_segments[21], axis=0),
            np.flip(self.profile_segments[29], axis=0),
        ])
        inner_cowl_with_flap = np.array([
            np.concatenate([self.xl_core, self.xu_core[::-1]]),
            np.concatenate([self.yl_core, self.yu_core[::-1]]),
        ]).T
        outer_cowl_with_flap = np.array([
            np.concatenate([self.xl_bp, self.xu_bp[::-1]]),
            np.concatenate([self.yl_bp, self.yu_bp[::-1]]),
        ]).T
        
        ##
        
        for i, coords in enumerate([
            centre_body, rotor, outer_stator, hx, inner_stator, inner_cowl_with_flap, outer_cowl_with_flap
        ]):
            ax.fill(
                coords[:,0],
                coords[:,1],
                facecolor=color, alpha=alpha
            )
            ax.fill(
                coords[:,0],
                -coords[:,1],
                facecolor=color, alpha=alpha
            )
        
        
        # ax.set_xlim(-1, 3)
        # ax.set_ylim(0, 2)
            
        if show:
            plt.show()
        
        
    def plot_frontal_view(self, fig=None, ax=None, show=True, color='k', alpha=0.25) -> None:
        '''
        Plot frontal view.
        '''
        
        if (fig == None or ax == None):
            fig, ax = plt.subplots(figsize=(16, 8))
        
        r_tip_rotor = self.params['r_tip_rotor']
        r_hub_rotor = self.params['r_hub_rotor']
        r_intake_highlight = self.segment_points[8][1]
        r_inner_cowl_highlight = self.segment_points[5][1]
        
        r_in_spiral = self.params['r_in_spiral']
        t_out_spiral = self.params['t_out_spiral']
        N_spirals_t = self.params['N_spirals_t']
    
        def X(r, t, a):
            return r * (np.cos(t + a) + t * np.sin(t + a))
        def Y(r, t, a):
            return r * (np.sin(t + a) - t * np.cos(t + a))
        
        t_array = np.linspace(0, t_out_spiral, 50)
        a_array = np.linspace(0, 2 * np.pi, int(N_spirals_t))
    
        # Data  digitised from Figure 2c of medium_paper_2022_uhbropentestcasefanecl5CATANA_pages using WebPlotDigitizer
        blade_data = np.array([
            [0.017455833368469564, 0.2814858540287558],
            [0.020942783657291963, 0.3352995741451281],
            [0.022724908996359927, 0.3849713988559543],
            [0.02537420415735525, 0.4454062486823799],
            [0.027159140419670935, 0.4967337071861254],
            [0.032349510196623354, 0.5538615058115841],
            [0.04347092802631014, 0.6043766074967325],
            [0.06393163834661508, 0.6557349861562032],
            [0.0903191803347807, 0.697997217185985],
            [0.1243454062486824, 0.7394442804739219],
            [0.15922474736827308, 0.7833761999128616],
            [0.18815758035726815, 0.8248148304310552],
            [0.21030203370296952, 0.8678978510491773],
            [0.22140799145479328, 0.9093069668732697],
            [0.22401652822869678, 0.9457351267023656],
            [0.15952692161740534, 0.9613568326516846],
            [0.08484209638655815, 0.9719947716827594],
            [0.005901533358631417, 0.9760031482340377],
            [-0.07135247572065051, 0.9733918005368866],
            [-0.06377141572149392, 0.9386361400401964],
            [-0.06214670208429984, 0.8955924723475428],
            [-0.06562943598825055, 0.8442622029205494],
            [-0.07335525853466573, 0.7937527230818959],
            [-0.08023780410675885, 0.7399333811190287],
            [-0.08287023372826818, 0.6894323340501192],
            [-0.07787241219378549, 0.6331492178606064],
            [-0.06607918370788912, 0.5793607960534639],
            [-0.05512501581144291, 0.5313656870599158],
            [-0.040776658093350404, 0.48254838301640246],
            [-0.02897218591446382, 0.4353824963809364],
            [-0.01734901828505686, 0.2814282301021772],
            [-0.019802954280333296, 0.33605992888364267],
            [-0.02311422186617218, 0.38572332082472505],
        ])
        
        def rotate(points: np.ndarray, alpha: float) -> np.ndarray:
            """Rotate Nx2 array `points` about the origin by angle `alpha` (radians)."""
            c, s = np.cos(alpha), np.sin(alpha)
            return points.dot([[c, -s], [s, c]])
        
        f_scale = r_tip_rotor / np.max(blade_data[:,1])
        
        circle0 = plt.Circle((0, 0), r_hub_rotor, facecolor=color, edgecolor=color, fill=True, alpha=alpha)
        circle1 = plt.Circle((0, 0), r_intake_highlight, facecolor=None, edgecolor=color, fill=False, alpha=alpha)
        circle2 = plt.Circle((0, 0), r_tip_rotor, facecolor=None, edgecolor=color, fill=False, alpha=alpha)
        circle3 = plt.Circle((0, 0), r_inner_cowl_highlight, facecolor=None, edgecolor=color, fill=False, alpha=alpha)
        
        ax.add_patch(circle0)
        ax.add_patch(circle1)
        # ax.add_patch(circle2)
        # ax.add_patch(circle3)
        
        for a in a_array: # only plot every tenth spiral for better visibility
            x = X(r_in_spiral, t_array, a)
            y = Y(r_in_spiral, t_array, a)
            plt.plot(x, y, color=color, linewidth=0.5, alpha=alpha)
            
        angular_range = np.linspace(0, 2 * np.pi, 16)
        scaled_blade_data = f_scale * blade_data
        for angular in angular_range:
            rotated_scaled_blade_data = rotate(scaled_blade_data, angular)
            ax.plot(rotated_scaled_blade_data[:,0], rotated_scaled_blade_data[:,1], color=color, zorder=-1, linewidth=1, alpha=alpha)
            
        ax.set_aspect('equal')
        
        if show:
            plt.show()
        
        



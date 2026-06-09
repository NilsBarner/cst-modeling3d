'''
Jet engine by Long Yizhu (2023.01)
'''

import numpy as np
import pandas as pd
import math

# NILS: added to prevent `FileNotFoundError: [Errno 2] No such file or directory: './dump/engine_outward.dat'`
import sys
sys.path.append('.')
import matplotlib.pyplot as plt

from cst_modeling.basic import BasicSection, BasicSurface
from cst_modeling.section import cst_curve 
from cst_modeling.section import RoundTipSection


class Section(BasicSection):

    def __init__(self, n_curve = 4):
        self.x = np.zeros(1)
        self.y = np.zeros(1)
        self.z = np.zeros(1)
        self.n_ = max(1,n_curve)
        self.curve = [curve() for _ in range(self.n_)]

    def set_Section(self, sec_x, sec_y, sec_z):
        self.x = sec_x
        self.y = sec_y
        self.z = sec_z

    def join_curve(self):
        for i in range(self.n_):
            self.x = np.append(self.x, self.curve[i].curve_x)
            self.y = np.append(self.y, self.curve[i].curve_y)
            self.z = np.append(self.z, self.curve[i].curve_z)

        self.x = np.delete(self.x,0)
        self.y = np.delete(self.y,0)
        self.z = np.delete(self.z,0)

        self.x,self.y,self.z = order_curve3d(self.x,self.y,self.z)

        self.x[0] = self.x[-1]
        self.y[0] = self.y[-1]
        self.z[0] = self.z[-1]

    def circle(self, p1=np.array([1,0]), p2=np.array([0,1]), p3=np.array([0,-1]), nn=500):
        A = p1[0]*(p2[1]-p3[1]) - p1[1]*(p2[0]-p3[0]) + p2[0]*p3[1] - p3[0]*p2[1]
        if np.abs(A) <= 1E-20:
            raise Exception('Finding circle: 3 points in one line')
        
        p1s = p1[0]**2 + p1[1]**2
        p2s = p2[0]**2 + p2[1]**2
        p3s = p3[0]**2 + p3[1]**2

        B = p1s*(p3[1]-p2[1]) + p2s*(p1[1]-p3[1]) + p3s*(p2[1]-p1[1])
        C = p1s*(p2[0]-p3[0]) + p2s*(p3[0]-p1[0]) + p3s*(p1[0]-p2[0])
        D = p1s*(p3[0]*p2[1]-p2[0]*p3[1]) + p2s*(p1[0]*p3[1]-p3[0]*p1[1]) + p3s*(p2[0]*p1[1]-p1[0]*p2[1])

        cen_x = -B/2/A
        cen_y = -C/2/A

        R = np.sqrt(B**2+C**2-4*A*D)/2/np.abs(A)

        for i in range(nn):
            theta = i*2*np.pi/(nn-1)
            self.x = np.append(self.x, cen_x + R*math.cos(theta))
            self.y = np.append(self.y, cen_y + R*math.sin(theta))
        self.x = np.delete(self.x,0)
        self.y = np.delete(self.y,0)


    def order_curve3d(x,y,z,axis = 'z'):
        if axis == 'x':
            x1 = y
            x2 = z
            x3 = x
        if axis == 'y':
            x1 = x
            x2 = z
            x3 = y
        if axis == 'z':  
            x1 = x
            x2 = y
            x3 = z

        cen_x = np.mean(x1)
        cen_y = np.mean(x2)
        
        x1s = []
        x2s = []
        x3s = []
        for i in range(len(x1)):
            dx = x1[i] - cen_x
            dy = x2[i] - cen_y
            ang = np.arctan2(dy,dx)
            if ang < 0:
                ang += np.pi*2
            x1s.append([x1[i],ang])
            x2s.append([x2[i],ang])
            x3s.append([x3[i],ang])

        x1s = sorted(x1s, key = lambda x:x[1])
        x2s = sorted(x2s, key = lambda x:x[1])
        x3s = sorted(x3s, key = lambda x:x[1])

        x1 = np.array([x[0] for x in x1s])
        x2 = np.array([x[0] for x in x2s])
        x3 = np.array([x[0] for x in x3s])

        if axis == 'x':
            return x3,x1,x2
        if axis == 'y':
            return x1,x3,x2
        if axis == 'z':  
            return x1,x2,x3

class curve():

    def __init__(self, r0=np.array([0,0,0]), r1=np.array([1,0,0]), normal = np.array([0,0,1]), nn=125, coef = np.array([0,0,0,0]), xn1=0.75, xn2=1.0):
        #确定曲线增加的方向
        dr = r1 - r0
        D = np.cross(normal,dr)
        D = D / np.linalg.norm(D)

        #确定基本直线
        L0 = np.linalg.norm(dr)
        base_x = np.zeros(nn)
        base_y = np.zeros(nn)
        base_z = np.zeros(nn)
        for i in range(nn):
            tt = i / (nn-1)
            base_x[i] = (1-tt)*r0[0] + tt*r1[0]
            base_y[i] = (1-tt)*r0[1] + tt*r1[1]
            base_z[i] = (1-tt)*r0[2] + tt*r1[2]

        #生成cst曲线
        cst_x,cst_y = cst_curve(nn,coef,None,xn1,xn2)
        cst_y = cst_y * L0

        self.curve_x = base_x + D[0]*cst_y
        self.curve_y = base_y + D[1]*cst_y
        self.curve_z = base_z + D[2]*cst_y

def read_sec_file(filename,scale = 1):
    
    df = pd.read_excel(filename,header=0)
    data = df.values
    z = data[1:,1]
    y = data[1:,2]
    x = data[1:,3]

    x,y,z = order_curve3d(x,y,z)

    x = np.append(x,x[0])
    y = np.append(y,y[0])
    z = np.append(z,z[0])

    return x*scale,y*scale,z*scale

def order_curve3d(x,y,z,axis = 'z'):
    
    if axis == 'x':
        x1 = y
        x2 = z
        x3 = x
    if axis == 'y':
        x1 = x
        x2 = z
        x3 = y
    if axis == 'z':  
        x1 = x
        x2 = y
        x3 = z

    cen_x = np.mean(x1)
    cen_y = np.mean(x2)
    
    x1s = []
    x2s = []
    x3s = []
    for i in range(len(x1)):
        dx = x1[i] - cen_x
        dy = x2[i] - cen_y
        ang = np.arctan2(dy,dx)
        if ang < 0:
            ang += np.pi*2
        x1s.append([x1[i],ang])
        x2s.append([x2[i],ang])
        x3s.append([x3[i],ang])

    x1s = sorted(x1s, key = lambda x:x[1])
    x2s = sorted(x2s, key = lambda x:x[1])
    x3s = sorted(x3s, key = lambda x:x[1])

    x1 = np.array([x[0] for x in x1s])
    x2 = np.array([x[0] for x in x2s])
    x3 = np.array([x[0] for x in x3s])

    if axis == 'x':
        return x3,x1,x2
    if axis == 'y':
        return x1,x3,x2
    if axis == 'z':  
        return x1,x2,x3

#%%

if __name__== "__main__":
    
    # Intake
    dz_hx_intake = -0.5
    w_intake = 0.75
    h_intake = 0.25
    # Upstream duct
    l_up_duct = 1
    # HX
    l_hx = 1.5
    w_hx = 1
    h_hx = 0.75
    dx_hx_corner = 0.5
    dz_hx_corner = 0.15
    # Downstream duct
    l_down_duct = 0.5
    # Fan
    dz_hx_fan = 0.5
    d_fan = 0.5
    l_fan = 0.3
    # Nozzle
    d_nozzle = 0.3
    l_nozzle = 0.3
    
    x_centroid_hx = 0.0
    y_centroid_hx = 0.0
    z_centroid_hx = 0.0
    
    inward = BasicSurface(n_sec=6, name='out', nn=201, ns=51, projection = False)
    
    ### NILS: intake
    
    x_centroid_intake = x_centroid_hx - l_hx/2 - l_up_duct
    y_centroid_intake = y_centroid_hx
    z_centroid_intake = z_centroid_hx + dz_hx_intake
    
    # Define intake geometry
    xx = np.linspace(0, 1, 250)  # NILS: number x2 must match numbers in `curve()` calls x4
    x_, y_ = RoundTipSection.base_shape(
        xx, x_LE=0, x_TE=w_intake, l_LE=w_intake/10, l_TE=w_intake/10,
        r_LE=w_intake/10, r_TE=w_intake/10, h=h_intake/2, i_split=None,
    )
    
    # Create 2D profile
    upper_sec = BasicSection(thick=None, chord=1.0, twist=0.0, lTwistAroundLE=False)
    upper_sec.xx = x_
    upper_sec.yy = y_
    upper_sec.xLE = x_centroid_intake  # 1.0
    upper_sec.yLE = -w_intake/2
    upper_sec.zLE = z_centroid_intake
    
    lower_sec = BasicSection(thick=None, chord=1.0, twist=0.0, lTwistAroundLE=False)
    lower_sec.xx = x_
    lower_sec.yy = -y_
    lower_sec.xLE = x_centroid_intake  # 1.0
    lower_sec.yLE = -w_intake/2
    lower_sec.zLE = z_centroid_intake
    
    # NILS: create 3D section from 2D profile
    upper_sec.section(flip_x=False, projection=True)
    lower_sec.section(flip_x=False, projection=True)

    # Convert into `Section` in context of this script
    inward.secs[0] = Section(n_curve = 2)
    inward.secs[0].curve[0].curve_x = z_centroid_intake + y_
    inward.secs[0].curve[0].curve_y = -w_intake/2 + x_
    inward.secs[0].curve[0].curve_z = x_centroid_intake * np.ones_like(x_)
    inward.secs[0].curve[1].curve_x = z_centroid_intake - y_
    inward.secs[0].curve[1].curve_y = -w_intake/2 + x_
    inward.secs[0].curve[1].curve_z = x_centroid_intake * np.ones_like(x_)
    inward.secs[0].join_curve()
    
    # Plot intake outline
    
    # fig, ax = plt.subplots()
    # ax.scatter(upper_sec.x, upper_sec.y)
    # ax.scatter(lower_sec.x, lower_sec.y)
    # ax.set_aspect('equal')
    # plt.show()
    
    # fig, ax = plt.subplots()
    # ax.scatter(upper_sec.xx, upper_sec.yy)
    # ax.scatter(lower_sec.xx, lower_sec.yy)
    # ax.set_aspect('equal')
    # plt.show()
    
    ### NILS: upstream duct outlet
    inward.secs[1] = Section(n_curve = 4)
    inward.secs[1].curve[0] = curve(
        r0=np.array([z_centroid_hx - h_hx/2, y_centroid_hx + w_hx/2, x_centroid_hx + dx_hx_corner]),
        r1=np.array([z_centroid_hx + dz_hx_corner, y_centroid_hx + w_hx/2, x_centroid_hx - l_hx/2]),
        nn=125,
    )
    inward.secs[1].curve[1] = curve(
        r0=np.array([z_centroid_hx + dz_hx_corner, y_centroid_hx + w_hx/2, x_centroid_hx - l_hx/2]),
        r1=np.array([z_centroid_hx + dz_hx_corner, y_centroid_hx - w_hx/2, x_centroid_hx - l_hx/2]),
        nn=125,
    )
    inward.secs[1].curve[2] = curve(
        r0=np.array([z_centroid_hx + dz_hx_corner, y_centroid_hx - w_hx/2, x_centroid_hx - l_hx/2]),
        r1=np.array([z_centroid_hx - h_hx/2, y_centroid_hx - w_hx/2, x_centroid_hx + dx_hx_corner]),
        nn=125,
    )
    inward.secs[1].curve[3] = curve(
        r0=np.array([z_centroid_hx - h_hx/2, y_centroid_hx - w_hx/2, x_centroid_hx + dx_hx_corner]),
        r1=np.array([z_centroid_hx - h_hx/2, y_centroid_hx + w_hx/2, x_centroid_hx + dx_hx_corner]),
        nn=125,
    )
    inward.secs[1].join_curve()
    
    ### NILS: HX outlet
    inward.secs[2] = Section(n_curve = 4)
    inward.secs[2].curve[0] = curve(
        r0=np.array([z_centroid_hx - dz_hx_corner, y_centroid_hx + w_hx/2, x_centroid_hx + l_hx/2]),
        r1=np.array([z_centroid_hx + h_hx/2, y_centroid_hx + w_hx/2, x_centroid_hx - dx_hx_corner]),
        nn=125,
    )
    inward.secs[2].curve[1] = curve(
        r0=np.array([z_centroid_hx + h_hx/2, y_centroid_hx + w_hx/2, x_centroid_hx - dx_hx_corner]),
        r1=np.array([z_centroid_hx + h_hx/2, y_centroid_hx - w_hx/2, x_centroid_hx - dx_hx_corner]),
        nn=125,
    )
    inward.secs[2].curve[2] = curve(
        r0=np.array([z_centroid_hx + h_hx/2, y_centroid_hx - w_hx/2, x_centroid_hx - dx_hx_corner]),
        r1=np.array([z_centroid_hx - dz_hx_corner, y_centroid_hx - w_hx/2, x_centroid_hx + l_hx/2]),
        nn=125,
    )
    inward.secs[2].curve[3] = curve(
        r0=np.array([z_centroid_hx - dz_hx_corner, y_centroid_hx - w_hx/2, x_centroid_hx + l_hx/2]),
        r1=np.array([z_centroid_hx - dz_hx_corner, y_centroid_hx + w_hx/2, x_centroid_hx + l_hx/2]),
        nn=125,
    )
    inward.secs[2].join_curve()
    
    ### NILS: fan inlet
    
    x_centroid_fan_inlet = x_centroid_hx + l_hx/2 + l_down_duct
    y_centroid_fan_inlet = y_centroid_hx + dz_hx_fan  # NILS: note coordinate change
    z_centroid_fan_inlet = z_centroid_hx
    
    inward.secs[3] = Section(n_curve = 4)
    inward.secs[3].circle(
        np.array([y_centroid_fan_inlet - d_fan/2, z_centroid_fan_inlet]),
        np.array([y_centroid_fan_inlet + d_fan/2, z_centroid_fan_inlet]),
        np.array([y_centroid_fan_inlet, z_centroid_fan_inlet + d_fan/2]),
        nn=500,
    )
    inward.secs[3].z = x_centroid_fan_inlet * np.ones_like(inward.secs[3].x)
    #将曲线排序
    inward.secs[3].x, inward.secs[3].y, inward.secs[3].z = order_curve3d(
        inward.secs[3].x,
        inward.secs[3].y,
        inward.secs[3].z,
    )
    #让曲线封闭
    inward.secs[3].x[0] = inward.secs[3].x[-1]
    inward.secs[3].y[0] = inward.secs[3].y[-1]
    inward.secs[3].z[0] = inward.secs[3].z[-1]
    
    ### NILS: fan outlet
    
    x_centroid_fan_outlet = x_centroid_hx + l_hx/2 + l_down_duct + l_fan
    y_centroid_fan_outlet = y_centroid_hx + dz_hx_fan  # NILS: note coordinate change
    z_centroid_fan_outlet = z_centroid_hx
    
    inward.secs[4] = Section(n_curve = 4)
    inward.secs[4].circle(
        np.array([y_centroid_fan_outlet - d_fan/2, z_centroid_fan_outlet]),
        np.array([y_centroid_fan_outlet + d_fan/2, z_centroid_fan_outlet]),
        np.array([y_centroid_fan_outlet, z_centroid_fan_outlet + d_fan/2]),
        nn=500,
    )
    inward.secs[4].z = x_centroid_fan_outlet * np.ones_like(inward.secs[4].x)
    #将曲线排序
    inward.secs[4].x, inward.secs[4].y, inward.secs[4].z = order_curve3d(
        inward.secs[4].x,
        inward.secs[4].y,
        inward.secs[4].z,
    )
    #让曲线封闭
    inward.secs[4].x[0] = inward.secs[4].x[-1]
    inward.secs[4].y[0] = inward.secs[4].y[-1]
    inward.secs[4].z[0] = inward.secs[4].z[-1]
    
    ### NILS: nozzle outlet
    
    x_centroid_nozzle_outlet = x_centroid_hx + l_hx/2 + l_down_duct + l_fan + l_nozzle
    y_centroid_nozzle_outlet = y_centroid_hx + dz_hx_fan  # NILS: note coordinate change
    z_centroid_nozzle_outlet = z_centroid_hx
    
    inward.secs[5] = Section(n_curve = 4)
    inward.secs[5].circle(
        np.array([y_centroid_nozzle_outlet - d_nozzle/2, z_centroid_nozzle_outlet]),
        np.array([y_centroid_nozzle_outlet + d_nozzle/2, z_centroid_nozzle_outlet]),
        np.array([y_centroid_nozzle_outlet, z_centroid_nozzle_outlet + d_nozzle/2]),
        nn=500,
    )
    inward.secs[5].z = x_centroid_nozzle_outlet * np.ones_like(inward.secs[5].x)
    #将曲线排序
    inward.secs[5].x, inward.secs[5].y, inward.secs[5].z = order_curve3d(
        inward.secs[5].x,
        inward.secs[5].y,
        inward.secs[5].z,
    )
    #让曲线封闭
    inward.secs[5].x[0] = inward.secs[5].x[-1]
    inward.secs[5].y[0] = inward.secs[5].y[-1]
    inward.secs[5].z[0] = inward.secs[5].z[-1]
    
    # Plot sections
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(inward.secs[0].x, inward.secs[0].y, inward.secs[0].z)
    ax.scatter(inward.secs[1].x, inward.secs[1].y, inward.secs[1].z)
    ax.scatter(inward.secs[2].x, inward.secs[2].y, inward.secs[2].z)
    ax.scatter(inward.secs[3].x, inward.secs[3].y, inward.secs[3].z)
    ax.scatter(inward.secs[4].x, inward.secs[4].y, inward.secs[4].z)
    ax.scatter(inward.secs[5].x, inward.secs[5].y, inward.secs[5].z)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_aspect('equal')
    plt.show()
    
    # =============================================================================
    # NILS
    import numpy as np
    from copy import deepcopy
    
    from cst_modeling.basic import interp_basic_sec
    
    def rotate_section_y(sec, angle_deg):
        theta = np.radians(angle_deg)
        c, s = np.cos(theta), np.sin(theta)
    
        x = sec.x.copy()
        z = sec.z.copy()
    
        sec.x = c*x + s*z
        sec.z = -s*x + c*z
        
    
    # --- helper functions for 3D-only sections (drop-in replacement) ---
    import copy
    from scipy.interpolate import interp1d
    
    def resample_curve(x, y, z, n_points):
        """
        Resample a 3D closed or open curve (x,y,z) to exactly n_points (including closure if closed).
        Returns (x_new, y_new, z_new) as numpy arrays.
        - If input is closed (first==last within tol) output will be closed (last equals first).
        - For closed curves we sample n_points-1 around the loop and then append the first point to close.
        """
        x = np.asarray(x).astype(float)
        y = np.asarray(y).astype(float)
        z = np.asarray(z).astype(float)
    
        if x.size < 2:
            return x.copy(), y.copy(), z.copy()
    
        # Detect closure (first and last coincident)
        closed = np.allclose([x[0], y[0], z[0]], [x[-1], y[-1], z[-1]], atol=1e-9)
    
        if closed and x.size > 1:
            # drop duplicate last point for interpolation
            x0 = x[:-1]
            y0 = y[:-1]
            z0 = z[:-1]
        else:
            x0 = x
            y0 = y
            z0 = z
    
        # arc-length along the curve
        dx = np.diff(x0)
        dy = np.diff(y0)
        dz = np.diff(z0)
        seg = np.sqrt(dx*dx + dy*dy + dz*dz)
        s = np.concatenate(([0.0], np.cumsum(seg)))
        total_len = s[-1]
        if total_len <= 0:
            # degenerate - return replicated points
            xp = np.linspace(0.0, 1.0, n_points)
            return np.interp(xp, [0.0,1.0], [x0[0], x0[-1]]), \
                   np.interp(xp, [0.0,1.0], [y0[0], y0[-1]]), \
                   np.interp(xp, [0.0,1.0], [z0[0], z0[-1]])
    
        # sampling parameter
        if closed:
            # for closed keep first point duplicated at the end: sample n_points-1 on [0,total_len)
            t_new = np.linspace(0.0, total_len, num=max(2, n_points-1), endpoint=False)
        else:
            t_new = np.linspace(0.0, total_len, num=n_points, endpoint=True)
    
        fx = interp1d(s, x0, kind='linear', bounds_error=False, fill_value='extrapolate')
        fy = interp1d(s, y0, kind='linear', bounds_error=False, fill_value='extrapolate')
        fz = interp1d(s, z0, kind='linear', bounds_error=False, fill_value='extrapolate')
    
        x_new = fx(t_new)
        y_new = fy(t_new)
        z_new = fz(t_new)
    
        if closed:
            # append first point again to close curve
            x_new = np.append(x_new, x_new[0])
            y_new = np.append(y_new, y_new[0])
            z_new = np.append(z_new, z_new[0])
    
        return x_new, y_new, z_new
    
    
    def insert_intermediate_sections_3d(surface, i0, n_mid=2, n_points=None):
        """
        Insert n_mid intermediate sections between surface.secs[i0] and surface.secs[i0+1].
        Interpolation is done directly on the 3D coordinates (x,y,z).
        The inserted sections keep the same python class as surface.secs[i0] (via deepcopy).
        After insertion, call surface.geo(update_sec=False) to rebuild surfaces.
    
        Parameters
        ----------
        surface : BasicSurface (or your surface object)
        i0 : int
            index of left section
        n_mid : int
            number of intermediate sections to insert
        n_points : int or None
            number of points for re-sampling each section curve; if None use max(len(sec0), len(sec1))
        """
        if n_mid <= 0:
            return
    
        sec0 = surface.secs[i0]
        sec1 = surface.secs[i0 + 1]
    
        # sanity checks
        if not (hasattr(sec0, 'x') and hasattr(sec0, 'y') and hasattr(sec0, 'z')):
            raise RuntimeError("insert_intermediate_sections_3d: sec0 missing x/y/z arrays")
        if not (hasattr(sec1, 'x') and hasattr(sec1, 'y') and hasattr(sec1, 'z')):
            raise RuntimeError("insert_intermediate_sections_3d: sec1 missing x/y/z arrays")
    
        n0 = int(np.asarray(sec0.x).size)
        n1 = int(np.asarray(sec1.x).size)
        target_n = int(n_points) if n_points is not None else max(n0, n1, 4)
    
        x0, y0, z0 = resample_curve(sec0.x, sec0.y, sec0.z, target_n)
        x1, y1, z1 = resample_curve(sec1.x, sec1.y, sec1.z, target_n)
    
        # create and insert new sections
        for k in range(n_mid):
            r = (k + 1) / (n_mid + 1.0)
            new_sec = copy.deepcopy(sec0)     # keep same class & metadata, but override coords
            new_sec.x = (1.0 - r) * x0 + r * x1
            new_sec.y = (1.0 - r) * y0 + r * y1
            new_sec.z = (1.0 - r) * z0 + r * z1
            # if the class has e.g. .curve list it will be preserved (OK); coords replaced
            surface.secs.insert(i0 + 1 + k, new_sec)
    
    
    def estimate_dyn0(surface, i0, i1):
        """
        same as your earlier estimator but robust to small arrays
        """
        sec0 = surface.secs[i0]
        sec1 = surface.secs[i1]
        y0_mean = np.mean(sec0.y) if hasattr(sec0, 'y') else 0.0
        z0_mean = np.mean(sec0.z) if hasattr(sec0, 'z') else 0.0
        y1_mean = np.mean(sec1.y) if hasattr(sec1, 'y') else 0.0
        z1_mean = np.mean(sec1.z) if hasattr(sec1, 'z') else 0.0
        dz = (z1_mean - z0_mean)
        if abs(dz) < 1e-9:
            return 0.0
        return (y1_mean - y0_mean) / dz
    
    
    def improve_transition(surface, i_sec0, n_mid=3, ratio_end=[5,5,1.2], dyn0=None,
                           rotate_start_deg=None, rotate_end_deg=None, n_resample=None):
        """
        High-level routine to improve the transition between section i_sec0 and i_sec0+1.
        - Rotates the end sections (about Y) in-place if requested (rotation BEFORE interpolation).
        - Inserts n_mid intermediate sections (via insert_intermediate_sections_3d).
        - Rebuilds surface.surfs using surface.geo(update_sec=False).
        - Calls surface.smooth(...) over the extended block.
    
        n_resample: optional, pass integer to control number of points per section after resample.
        """
        i0 = i_sec0
        i1 = i_sec0 + 1
    
        # rotate end sections in-place (if requested)
        if rotate_start_deg is not None:
            rotate_section_y(surface.secs[i0], rotate_start_deg)
    
        if rotate_end_deg is not None:
            rotate_section_y(surface.secs[i1], rotate_end_deg)
    
        # insert intermediate sections (works purely on sec.x,y,z)
        insert_intermediate_sections_3d(surface, i0, n_mid=n_mid, n_points=n_resample)
    
        new_i1 = i1 + n_mid
    
        # rebuild surface geometry from existing secs.x,y,z
        surface.geo(update_sec=False)
    
        # prepare dyn0 if not provided
        if dyn0 is None:
            dyn0 = estimate_dyn0(surface, i0, new_i1)
    
        # call smooth over the new block (smooth modifies surface.surfs in-place)
        surface.smooth(i_sec0=i0, i_sec1=new_i1, smooth0=False, smooth1=False, dyn0=dyn0, ratio_end=ratio_end)
    
        return
    # --- end helper functions ---
    
    # # =============================================================================
    # # rebuild surface geometry from existing secs.x,y,z
    # inward.geo(update_sec=False)
    # # =============================================================================
    
    improve_transition(inward, i_sec0=0, n_mid=4, ratio_end=[5,5,1.2], dyn0=None, rotate_end_deg=15.0)  # NILS: change `n_mid`
    improve_transition(inward, i_sec0=2, n_mid=4, ratio_end=[5,5,1.2], dyn0=None, rotate_end_deg=0.0)  # NILS: change `n_mid`
    
    # you can tune n_mid (2..6), rotate_end_deg to match the exact rotation you want,
    # and ratio_end to change tightness (bigger values tighten shape near ends).
    
    # After both transitions were improved, you may want to flip/orient and then write file:
    inward.flip(axis = '+Y')
    inward.flip(axis = '+X')
    inward.output_tecplot(fname='engine_inward_new_smoothed.dat')
    # =============================================================================
    '''
    # Convert to mesh
    
    inward.geo(update_sec = False)
    
    inward.smooth(0,1,smooth0=True,ratio_end=-1)
    inward.smooth(2,3,smooth0=True,ratio_end=-1)
    # inward.smooth(0,1,smooth0=True,ratio_end=10)
    # inward.smooth(2,3,smooth0=True,ratio_end=10)

    inward.flip(axis = '+Y')
    inward.flip(axis = '+X')

    inward.output_tecplot(fname='engine_inward_new.dat')  # NILS
    '''
    

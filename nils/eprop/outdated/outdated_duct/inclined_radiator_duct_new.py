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
        
    
    def insert_intermediate_sections(surface, i0, n_mid=2, axis='Z'):
        """
        Insert n_mid equally spaced intermediate sections between surface.secs[i0] and surface.secs[i0+1].
        Uses interp_basic_sec() (already in basic.py). After this function returns, surface.secs
        will contain the new sections inserted immediately after i0.
    
        NOTE: This function expects that surface.secs[*].xx/.yy (2D) and surface.secs[*].x/.y/.z (3D)
        are already present (i.e. call surface.update_sections() before calling this).
        """
        if n_mid <= 0:
            return
    
        # we will insert after i0, one-by-one, higher-index insert shifts next insertion position
        for k in range(n_mid):
            ratio = (k+1) / (n_mid + 1.0)
            sec_new = interp_basic_sec(surface.secs[i0], surface.secs[i0+1], ratio=ratio)
            # insert into slice (position i0+1 + k)
            surface.secs.insert(i0 + 1 + k, sec_new)
    
    
    def estimate_dyn0(surface, i0, i1):
        """
        Estimate a global dy/dz slope (single float) at the junction between the block of
        sections from i0..i1 (inclusive) used by BasicSurface.smooth(...) as dyn0.
        We compute a simple finite-difference slope between the averaged Y and Z coords
        of the two end control sections.
        """
        sec0 = surface.secs[i0]
        sec1 = surface.secs[i1]
        # use mean y & mean z of the entire section curve as a robust approximate anchor
        y0_mean = np.mean(sec0.y)
        z0_mean = np.mean(sec0.z)
        y1_mean = np.mean(sec1.y)
        z1_mean = np.mean(sec1.z)
        dz = (z1_mean - z0_mean)
        if abs(dz) < 1e-9:
            return 0.0
        return (y1_mean - y0_mean) / dz
    
    
    def improve_transition(surface, i_sec0, n_mid=3, ratio_end=[5,5,1.2], dyn0=None,
                           rotate_start_deg=None, rotate_end_deg=None):
        """
        High-level routine to improve the transition between section i_sec0 and i_sec0+1.
        - Inserts n_mid intermediate sections (via interpolation).
        - Optionally sets rotation about Y for the start and end sections (degrees).
        - Rebuilds surface geometry and calls smooth() over the expanded interval using parameters
          tuned to avoid overshoot and give smooth joins.
        """
    
        # # 1) ensure sections have 3D coords (compute if necessary)
        # surface.update_sections()
    
        i0 = i_sec0
        i1 = i_sec0 + 1
    
        # 2) optionally set section-level rotations (rot_y) on the two edge sections
        if rotate_start_deg is not None:
            surface.secs[i0].rot_y = rotate_start_deg
            # # recalc this section's 3D coords from its 2D profile and new rot_y
            # surface.secs[i0].section(nn=surface.nn, flip_x=False, projection=surface.projection)
            # =============================================================================
            rotate_section_y(surface.secs[i0], rotate_start_deg)
            # =============================================================================
    
        if rotate_end_deg is not None:
            surface.secs[i1].rot_y = rotate_end_deg
            # surface.secs[i1].section(nn=surface.nn, flip_x=False, projection=surface.projection)
            # =============================================================================
            rotate_section_y(surface.secs[i1], rotate_end_deg)
            # =============================================================================
    
        # 3) insert intermediate sections (they are created by linear interpolation in 3D)
        insert_intermediate_sections(surface, i0, n_mid=n_mid, axis='Z')
    
        # the new i1 index moves right by n_mid
        new_i1 = i1 + n_mid
    
        # 4) rebuild the surface geometry from current secs.x,y,z (we do NOT re-run update_sections
        #    so that we keep the interpolated sec.x,y,z exactly as created by interp_basic_sec)
        surface.geo(update_sec=False)
    
        # 5) prepare dyn0: if not provided, estimate from the two end controls (coarse but usually OK)
        if dyn0 is None:
            dyn0 = estimate_dyn0(surface, i0, new_i1)
    
        # 6) call smooth over the block that now spans i0..new_i1
        #    NOTE: smooth() takes i_sec0,i_sec1 such that it smooths surfaces i_sec0 ... i_sec1-1
        surface.smooth(i_sec0=i0, i_sec1=new_i1, smooth0=False, smooth1=False, dyn0=dyn0, ratio_end=ratio_end)
    
        # 7) done — surface.surfs have been updated and smoothed
        return
    
    
    # # build/update 3D control sections for all secs
    # inward.update_sections()   # or update only affected secs, but this is simplest
    
    # Improve transition between sections 0 and 1, insert 3 intermediates, rotate end by 15 degrees
    improve_transition(inward, i_sec0=0, n_mid=3, ratio_end=[5,5,1.2], dyn0=None, rotate_start_deg=None, rotate_end_deg=15.0)
    
    # Improve transition between sections 2 and 3, insert 3 intermediates, rotate end by 0.. adjust angle as needed
    improve_transition(inward, i_sec0=2, n_mid=3, ratio_end=[5,5,1.2], dyn0=None, rotate_start_deg=None, rotate_end_deg=0.0)
    
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
    

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
    
    # Convert to mesh
    
    inward.geo(update_sec = False)
    
    inward.smooth(0,1,smooth0=True,ratio_end=-1)
    inward.smooth(2,3,smooth0=True,ratio_end=-1)
    # inward.smooth(0,1,smooth0=True,ratio_end=10)
    # inward.smooth(2,3,smooth0=True,ratio_end=10)

    inward.flip(axis = '+Y')
    inward.flip(axis = '+X')

    inward.output_tecplot(fname='engine_inward_new.dat')  # NILS

    

"""
This script has been downloaded from
https://cst-modeling3d.readthedocs.io/en/latest/source/example/airfoil.html#extract-geometric-features
and shows how to modify airfoil geometry using the CST Modeling GitHub repo.
"""

__all__ = ["generate_fairing_geometry"]

import os
import sys
import copy
import numpy as np
import pyvista as pv
from matplotlib import pyplot as plt

from cst_modeling.section import cst_foil
from cst_modeling.foil import FoilGeoFeatures, FoilModification
from cst_modeling.basic import BasicSection, BasicSurface


def generate_fairing_geometry(
        l,
        t,
        h,
        dx=0.0,
        dy=0.0,
        dz=0.0,
    ):

    #* Initialize an airfoil
    cst_u = np.array([ 0.118598,  0.118914,  0.155731,  0.136732,  0.209265,  0.148305,  0.193591])
    cst_l = np.array([-0.115514, -0.134195, -0.109145, -0.253206, -0.012220, -0.118463,  0.064100])
    
    tail = 0.0
    rLE = 0.015
    width_bump = 1.0
    
    print('t/l =', t/l)
    x, yu, _, t0, rLE_old = cst_foil(1001, cst_u, cst_l, x=None, t=t/l, tail=tail)
    yl = -yu
    
    geo_old = FoilGeoFeatures(x, yu, yl)
    
    # plt.figure(figsize=(16,8))
    
    # plt.plot(x, geo_old.get_feature('thickness'), 'k--', lw=0.5)
    # plt.plot(x, -geo_old.get_feature('thickness'), 'k--', lw=0.5)
    
    # plt.xlim((-0.2, 1.2))
    # plt.ylim((-0.2, 0.2))
    # plt.axis('equal')
         
    #%%
    
    surface = BasicSurface(n_sec=2, name='fairing', nn=201, ns=51, projection=False)
    
    section = BasicSection()
    section.xx = x * l + dx
    section.yu = yu * l + dy
    # print('section.yu =', np.max(section.yu))
    section.yl = yl * l + dy
    
    section.section()
    
    section.z = dz
    
    section_2 = copy.deepcopy(section)
    # print('h =', h)
    section_2.z = dz + np.ones_like(section.z) * h
    
    print(section.z)
    print()
    print(section_2.z)
    
    surface.secs[0] = section
    surface.secs[1] = section_2
    
    surface.geo(update_sec=False)
    
    surface.flip(axis = '+Y')
    surface.flip(axis = '+X')
    
    mesh = pv.StructuredGrid(*surface.surfs[0])
    
    # Flip y- and z-axes to match convention
    mesh_flipped = mesh.copy()
    # mesh_flipped.points[:, [1, 2]] = mesh_flipped.points[:, [2, 1]]
    mesh_flipped.points = mesh_flipped.points[:, [1, 2, 0]]
    
    return mesh_flipped

#%%

if __name__ == '__main__':
    
    l = 0.5
    t = 0.15 * 0.5
    h = 2
    
    mesh = generate_fairing_geometry(
        l,
        t,
        h,
        dx=0.0,
        dy=0.0,
        dz=0.0,
    )
    
    plotter = pv.Plotter()
    plotter.add_mesh(mesh)
    plotter.add_axes()
    plotter.show_grid()
    plotter.show()
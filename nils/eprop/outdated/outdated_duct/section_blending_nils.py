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
from cst_modeling.section import Section
from cst_modeling.section import RoundTipSection, cst_foil_fit

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
xx = np.linspace(0, 1, 1000)  # NILS: number x2 must match numbers in `curve()` calls x4
x_, y_ = RoundTipSection.base_shape(
    xx, x_LE=0, x_TE=w_intake, l_LE=w_intake/10, l_TE=w_intake/10,
    # xx, x_LE=0, x_TE=w_intake, l_LE=1e-10, l_TE=1e-10,
    r_LE=w_intake/10, r_TE=w_intake/10, h=h_intake/2, i_split=None,
    # r_LE=0.0, r_TE=0.0, h=h_intake/2, i_split=None,
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

# Plot intake outline

fig, ax = plt.subplots()
ax.plot(upper_sec.x, upper_sec.y)
ax.plot(lower_sec.x, lower_sec.y)
ax.set_aspect('equal')
plt.show()

fig, ax = plt.subplots()
ax.plot(upper_sec.xx, upper_sec.yy)
ax.plot(lower_sec.xx, lower_sec.yy)
ax.set_aspect('equal')
plt.show()

cst_u, cst_l = cst_foil_fit(x_, y_, x_, -y_)
print('cst_u, cst_l =', cst_u, cst_l)

cst_section = Section(thick=None, chord=1.0, twist=0.0, lTwistAroundLE=False)
cst_section.section(cst_u, cst_l, nn=1001, flip_x=False, projection=True)

fig, ax = plt.subplots()
ax.plot(cst_section.xx, cst_section.yu)
ax.plot(cst_section.xx, cst_section.yl)
ax.set_aspect('equal')
plt.show()
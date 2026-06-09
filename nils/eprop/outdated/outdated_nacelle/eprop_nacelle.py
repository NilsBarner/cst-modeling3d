import sys
import numpy as np
from scipy.optimize import brentq
from math import comb

from flydrogen.geometry.parametric_geometry import generate_streamlined_body_geometry
# from flydrogen.geometry.parametric_geometry_new import generate_streamlined_body_geometry
from flydrogen.utilities.matplotlib_custom_settings import *


def define_quadratic(x0, y0, x1, y1):
    """
    Generate second-order polynomial tangent to point (x0, y0)
    and secant through point (x1, y1).
    """
    a = (y1 - y0) / (x1 - x0)**2
    b = -2 * a * x0
    c = y0 + a * x0**2
    return np.poly1d([a, b, c])

N_1, N_2 = 0.5, 1 # for round-nose airfoil
zeta_T = 0
Psi_Z_max = 0.35
S_Z_max = 0.15 # not provided but matches Fig. 2 reasonably well
Psi_le = 0
Psi_te = 1

S_le = 0.05 * 3  # NILS
S_te = 0.04 * 5  # NILS
Psi_range = np.linspace(1e-4, 1 - 1e-4, 50)

# Vary LE shape function
S_list = []
zeta_list = []
Psi_list = []

quadratic_front = define_quadratic(Psi_Z_max, S_Z_max, Psi_le, S_le)
# =============================================================================
quadratic_aft = define_quadratic(Psi_Z_max, S_Z_max, Psi_te, S_te)
# =============================================================================
for Psi in Psi_range:       
    if Psi >= Psi_Z_max:
        # =============================================================================
        # S = S_Z_max
        S = quadratic_aft(Psi)
        # =============================================================================
    elif Psi < Psi_Z_max:
        S = quadratic_front(Psi)
    
    C = Psi**N_1 * (1 - Psi)**N_2  # (6) / (31)
    zeta = C * S + Psi * zeta_T  # (7)
    
    S_list.append(S)
    zeta_list.append(zeta * 5)

# w = 2
# h = 2

# N_u = 0.25
# N_l = 0.05
N_u = 0.4
N_l = 0.25

def calc_coordinates(N_u, N_l):
    
    eta_range = np.linspace(0, 1, 100)
    
    y_u_list = []
    y_l_list = []
    z_u_list = []
    z_l_list = []
    
    for eta in eta_range:
        
        y = eta - 0.5
        
        zeta_u = 2 * eta**N_u * (1 - eta)**N_u
        zeta_l = 2 * eta**N_l * (1 - eta)**N_l
        
        zeta_u_max = 2 * (0.5**N_u) * (0.5**N_u)  # positive peak for upper lobe
        zeta_l_max = 2 * (0.5**N_l) * (0.5**N_l)  # magnitude for lower lobe
        
        zeta_u_norm = zeta_u / zeta_u_max / 2
        zeta_l_norm = -zeta_l / zeta_l_max / 2
        
        y_u_list.append(y)
        y_l_list.append(y)
        z_u_list.append(zeta_u_norm)
        z_l_list.append(zeta_l_norm)
        
    return y_u_list, y_l_list, z_u_list, z_l_list


y_u_list, y_l_list, z_u_list, z_l_list = calc_coordinates(N_u, N_l)

x_array = []
y_u_array = []
y_l_array = []
z_u_array = []
z_l_array = []

for i, f_scale in enumerate(zeta_list):
    
    psi = Psi_range[i]
    
    x_array.append(np.ones_like(y_u_list) * psi)
    y_u_array.append(np.array(y_u_list) * f_scale)
    y_l_array.append(np.array(y_l_list) * f_scale)
    z_u_array.append(np.array(z_u_list) * f_scale)
    z_l_array.append(np.array(z_l_list) * f_scale)
    

x_array = np.array(x_array)
y_u_array = np.array(y_u_array)
y_l_array = np.array(y_l_array)
z_u_array = np.array(z_u_array)
z_l_array = np.array(z_l_array)


fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')    

ax.plot_surface(
    x_array, y_u_array, z_u_array,
    rstride=4, cstride=4,         # stride controls how dense the grid lines are
    edgecolor='gray',             # mesh color
    facecolor='lightblue',        # fill color
    alpha=0.8                     # slightly transparent
)

ax.plot_surface(
    x_array, y_l_array, z_l_array,
    rstride=4, cstride=4,         # stride controls how dense the grid lines are
    edgecolor='gray',             # mesh color
    facecolor='lightblue',        # fill color
    alpha=0.8                     # slightly transparent
)

ax.set_xlabel('Axial coordinate (x)')
ax.set_ylabel('Horizontal (y)')
ax.set_zlabel('Vertical (z)')
ax.set_title('3D Hollow Body: Cross‐Section = Ellipse(a(x), b(x))')

# Adjust viewing angle if you like:
ax.view_init(elev=20, azim=120)

ax.set_aspect('equal')

plt.show()

#%%

# Plot RHS of Fig. 2
    
fig, ax = plt.subplots()
ax.plot(Psi_range, S_list)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.2)
ax.set_xlabel(r'$\Psi=x/c$')
ax.set_ylabel('$S$')
plt.show()
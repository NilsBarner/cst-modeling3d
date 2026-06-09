"""
Loft a Kulfan-style hollow body:
 - separate cubic interpolation of width (w) and height (h) from inlet -> outlet
 - cubic tilt alpha(x) about local y-axis
 - loft along a curved centreline z_c(x) (constant y_center)
Fully self-contained.  Adjust user params in the "User parameters" block.
"""
import numpy as np
import matplotlib.pyplot as plt

# ------------------ User parameters ------------------
# Cross-section dimension endpoints (measured in a vertical x=const plane)
w_in = 0.6        # inlet width (y-direction total)
h_in = 0.6  # 0.25     # inlet height (z-direction total)

w_out = 1.0       # outlet width
h_out = 0.6       # outlet height

# Tilt (rotation about local y axis)
tilt_end_deg = 25.0    # final tilt angle at outlet (degrees). inlet tilt assumed 0.

# Centreline / loft path
y_center = 0.0     # constant y of centreline
inlet_z = 0.0      # z of inlet centroid (user-specified)
outlet_z = 0.4     # z of outlet centroid

# geometry and sampling
L = 2.0            # axial length
eta_count = 120
psi_count = 120

# smoothing toggles (use cubic smoothstep)
use_smooth = True

# -----------------------------------------------------

# Kulfan-esque cubic used in your script
cubic = lambda x: 0.5 * (1.001 + 2 * x**3 - 3 * x**2)
# Smoothstep cubic (zero slopes at ends) for interpolation
smooth_cubic = (lambda t: 3*t*t - 2*t*t*t) if use_smooth else (lambda t: t)

# grids
eta_range = np.linspace(0.0, 1.0, eta_count)
psi_range = np.linspace(0.0, 1.0, psi_count)
eta_grid, psi_grid = np.meshgrid(eta_range, psi_range, indexing='ij')

# intermediate arrays (raw Kulfan-style coordinates)
x_array = np.zeros_like(eta_grid)       # axial station per column (function of psi)
y_array = np.zeros_like(eta_grid)       # raw lateral template
z_pos_array = np.zeros_like(eta_grid)   # raw positive vertical template
z_neg_array = np.zeros_like(eta_grid)   # raw negative vertical template

Nd = 0.005   # parameter used in your original script

# Populate base templates (as in your Fig.26 reproduction)
for i_eta, eta in enumerate(eta_range):
    for j_psi, psi in enumerate(psi_range):
        Nc = cubic(psi)

        W = 1.0
        H = 1.0  # 0.5

        Sc = 0.5**(2 * Nc)
        Cc = eta**Nc * (1 - eta)**Nc

        Sd = 0.5**(2 * Nd)
        Cd = psi**Nd * (1 - psi)**Nd

        x = psi * L
        y = -(Sd * Cd) * (1 - 2 * eta) * W / 2.0
        z = Sd * Cd * (Sc * Cc) * H / 2.0

        x_array[i_eta, j_psi] = x
        y_array[i_eta, j_psi] = y
        z_pos_array[i_eta, j_psi] = z
        z_neg_array[i_eta, j_psi] = -z

# Normalize templates so they are "unit" extents
# lateral template normalized to [-1,1], vertical templates normalized to [0,1]
max_abs_y = np.max(np.abs(y_array)) if np.max(np.abs(y_array)) > 0 else 1.0
y_template = y_array / max_abs_y  # values ~[-1, 1]

max_zpos = np.max(z_pos_array) if np.max(z_pos_array) > 0 else 1.0
zpos_template = z_pos_array / max_zpos  # values in [0,1]
# z negative template is mirror of positive (we keep sign when scaling later)
zneg_template = z_neg_array / (-np.min(z_neg_array) if np.min(z_neg_array) < 0 else 1.0)  # values in [0,1] (positive)

# width and height interpolation along x (psi index)
psi_positions = psi_range * L
# compute w(x) and h(x) columnwise as smooth transitions
t_vals = psi_range
S_vals = smooth_cubic(t_vals)
w_cols = w_in + (w_out - w_in) * S_vals
h_cols = h_in + (h_out - h_in) * S_vals

# tilt alpha(x)
alpha_cols_rad = np.deg2rad( tilt_end_deg * S_vals )   # radians at each psi column

# centreline z variation z_c(x)
z_centerline_cols = inlet_z + (outlet_z - inlet_z) * S_vals

# prepare arrays for final rotated coordinates
rot_X_pos = np.zeros_like(x_array)
rot_Y_pos = np.zeros_like(y_template)
rot_Z_pos = np.zeros_like(zpos_template)

rot_X_neg = np.zeros_like(x_array)
rot_Y_neg = np.zeros_like(y_template)
rot_Z_neg = np.zeros_like(zpos_template)

# Loop and place/rotate each template point columnwise
for j in range(psi_count):
    w_j = w_cols[j]
    h_j = h_cols[j]
    alpha = alpha_cols_rad[j]
    zc = z_centerline_cols[j]
    yc = y_center
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    # scale templates column j
    y_col = y_template[:, j] * (w_j / 2.0)                 # lateral offsets in [-w/2, w/2]
    zpos_col = zpos_template[:, j] * (h_j / 2.0)           # positive vertical offsets [0, h/2]
    zneg_col = - (zpos_template[:, j] * (h_j / 2.0))       # negative vertical offsets [-h/2, 0]

    # base axial for column (same for all eta)
    x_col = x_array[:, j]

    # rotate about local y for each eta row
    # rotated dx = sin(alpha) * z_rel ; rotated z = cos(alpha) * z_rel
    dx_pos = sa * zpos_col
    zpos_r = ca * zpos_col
    Xpos = x_col + dx_pos
    Ypos = yc + y_col
    Zpos = zc + zpos_r

    dx_neg = sa * zneg_col
    zneg_r = ca * zneg_col
    Xneg = x_col + dx_neg
    Yneg = yc + y_col
    Zneg = zc + zneg_r

    rot_X_pos[:, j] = Xpos
    rot_Y_pos[:, j] = Ypos
    rot_Z_pos[:, j] = Zpos

    rot_X_neg[:, j] = Xneg
    rot_Y_neg[:, j] = Yneg
    rot_Z_neg[:, j] = Zneg

# Plot result
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection='3d')

# plot upper and lower surfaces
ax.plot_surface(rot_X_pos, rot_Y_pos, rot_Z_pos,
                rstride=3, cstride=3, linewidth=0.2, edgecolor='gray',
                facecolor='lightcyan', alpha=0.9)

ax.plot_surface(rot_X_neg, rot_Y_neg, rot_Z_neg,
                rstride=3, cstride=3, linewidth=0.2, edgecolor='gray',
                facecolor='lightcyan', alpha=0.9)

# plot centreline
xc_line = psi_positions
zc_line = z_centerline_cols
yc_line = np.ones_like(xc_line) * y_center
ax.plot(xc_line, yc_line, zc_line, color='k', linewidth=2, label='centreline')

ax.set_xlabel('X (axial)')
ax.set_ylabel('Y (lateral)')
ax.set_zlabel('Z (vertical)')
ax.set_title('Loft between (w_in,h_in) -> (w_out,h_out) with tilt α(x) and curved centreline')

# set view and aspect as reasonable
ax.view_init(elev=20, azim=120)
try:
    ax.set_box_aspect((np.ptp(rot_X_pos), np.ptp(rot_Y_pos), np.ptp(np.concatenate((rot_Z_pos.ravel(), rot_Z_neg.ravel())))))
except Exception:
    pass

plt.legend()
plt.tight_layout()
plt.show()
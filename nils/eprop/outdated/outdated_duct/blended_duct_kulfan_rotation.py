import numpy as np
import matplotlib.pyplot as plt

#%% Reproduce Fig. 26 (with added cross-section tilt)

# smoothing cubic (smoothstep) used for tilt profile: zero slope at 0 and 1
smooth_cubic = lambda t: 3*t**2 - 2*t**3

# original cubic used for Nc in your script (kept for the shape)
cubic = lambda x: 0.5 * (1.001 + 2 * x**3 - 3 * x**2)

eta_range = np.linspace(0, 1, 100)
psi_range = np.linspace(0, 1, 100)
eta_grid, psi_grid = np.meshgrid(eta_range, psi_range)
x_array = np.zeros_like(eta_grid)
y_array = np.zeros_like(eta_grid)
z_pos_array = np.zeros_like(eta_grid)
z_neg_array = np.zeros_like(eta_grid)
y_norm_array = np.zeros_like(eta_grid)
z_norm_pos_array = np.zeros_like(eta_grid)
z_norm_neg_array = np.zeros_like(eta_grid)

Nd = 0.005

for k, eta in enumerate(eta_range):

    for l, psi in enumerate(psi_range):

        Nc = cubic(psi)

        # L = W = H = 1
        L = 2
        W = 1
        H = 0.5

        Sc = 0.5**(2 * Nc)  # (28)
        Cc = eta**Nc * (1 - eta)**Nc  # (29)

        Sd = 0.5**(2 * Nd)  # (30)
        Cd = psi**Nd * (1 - psi)**Nd  # (31)

        x = psi * L  # (32)
        y = -(Sd * Cd) * (1 - 2 * eta) * W / 2  # (33)
        z = Sd * Cd * (Sc * Cc) * H / 2  # (34)

        x_array[k, l] = x
        y_array[k, l] = y
        z_pos_array[k, l] = z
        z_neg_array[k, l] = -z

for l in np.arange(1, len(psi_range) - 1):

    y_norm_array[:, l] = y_array[:, l] / np.max(y_array) / 2

    z_norm_pos_array[:, l] = z_pos_array[:, l] / np.max(z_pos_array[:, l]) / 2
    z_norm_neg_array[:, l] = -z_neg_array[:, l] / np.min(z_neg_array[:, l]) / 2


# --- TILT IMPLEMENTATION ---
# Add a tilt about local y-axis that varies smoothly from 0 to tilt_end_deg
# along the axial coordinate (psi). The tilt is applied to both positive and
# negative z lobes. The tilt rotates the relative vector [0, y, z] about y.
#
# Parameters:
tilt_end_deg = 45.0   # final tilt angle at outlet in degrees (change as needed)
use_smooth = True     # if True use smooth cubic; if False linear ramp

# Precompute rotated arrays (same shapes)
rot_x = np.zeros_like(x_array)
rot_y = np.zeros_like(y_norm_array)
rot_z_pos = np.zeros_like(z_norm_pos_array)
rot_z_neg = np.zeros_like(z_norm_neg_array)

# For each column (psi) compute local tilt angle and rotate every row point
n_psi = psi_range.shape[0]
for l, psi in enumerate(psi_range):
    t = psi  # normalized axial coordinate 0..1
    if use_smooth:
        s = smooth_cubic(t)
    else:
        s = t
    alpha = np.deg2rad(tilt_end_deg * s)  # radians

    # rotation about local y: applied to relative vector [0, y, z]
    # for a relative x=0:
    #   delta_x = sin(alpha) * z
    #   z' = cos(alpha) * z
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    for k in range(eta_range.shape[0]):
        base_x = x_array[k, l]         # axial station (centerline)
        yv = y_norm_array[k, l]
        zpos = z_norm_pos_array[k, l]
        zneg = z_norm_neg_array[k, l]

        # rotate positive lobe
        dx_pos = sa * zpos
        zpos_r = ca * zpos
        x_pos_r = base_x + dx_pos

        # rotate negative lobe
        dx_neg = sa * zneg
        zneg_r = ca * zneg
        x_neg_r = base_x + dx_neg

        # We store rotated positive surface into the pos arrays, and negative accordingly.
        # Note: y remains unchanged by rotation about y-axis
        rot_x[k, l] = x_pos_r         # (we will use this for surface plotting)
        rot_y[k, l] = yv
        rot_z_pos[k, l] = zpos_r
        rot_z_neg[k, l] = -zneg_r    # keep negative sign for lower surface in plotting

# NOTE:
# rot_x/rot_y/rot_z_pos form the grid for the upper surface; rot_x/rot_y/rot_z_neg
# form the grid for the lower surface. This keeps the structure of the result
# similar to your original (two symmetric lobes) but with tilted planes.

# Plot the two surfaces in 3D with the tilt applied
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

# upper (positive z) surface
ax.plot_surface(
    rot_x, rot_y, rot_z_pos,
    rstride=4, cstride=4,
    edgecolor='gray',
    facecolor='lightblue',
    alpha=0.8
)

# lower (negative z) surface
ax.plot_surface(
    rot_x, rot_y, rot_z_neg,
    rstride=4, cstride=4,
    edgecolor='gray',
    facecolor='lightblue',
    alpha=0.8
)

ax.set_xlabel('Axial coordinate (x)')
ax.set_ylabel('Horizontal (y)')
ax.set_zlabel('Vertical (z)')
ax.set_title(f'3D Hollow Body with Cross-Section Tilt (end tilt = {tilt_end_deg}°)')

# Adjust viewing
ax.view_init(elev=20, azim=120)
ax.set_box_aspect((np.ptp(rot_x), np.ptp(rot_y), np.ptp(np.concatenate((rot_z_pos.ravel(), rot_z_neg.ravel())))))
plt.tight_layout()
plt.show()
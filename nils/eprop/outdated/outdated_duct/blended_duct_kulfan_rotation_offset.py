"""
Fig.26 reproduction extended:
 - cross-section tilt alpha(x) (rotation about local y)
 - loft along a curved centreline (constant y_center, z_center(x) varies)
Fully self-contained.
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------- User parameters ----------
tilt_end_deg = 25.0      # final tilt (degrees) at outlet (alpha(L))
use_smooth_tilt = True   # if True use cubic smoothstep, otherwise linear ramp

y_center = 0.2           # constant y-coordinate of centreline (loft along y=y_center)
inlet_z = 0.0            # z coordinate of the inlet plane centroid (user-specified)
outlet_z = 0.4           # z coordinate at outlet (choose desired final elevation)
use_smooth_centerline = True  # smooth cubic ramp for centreline z variation

# grid density and geometry parameters
eta_count = 100
psi_count = 100
eta_range = np.linspace(0.0, 1.0, eta_count)
psi_range = np.linspace(0.0, 1.0, psi_count)

# Kulfan-style function used in your original script
cubic = lambda x: 0.5 * (1.001 + 2 * x**3 - 3 * x**2)

# smoothing cubic (smoothstep) used for tilt & centreline: zero slope at 0 and 1
smooth_cubic = lambda t: 3*t**2 - 2*t**3

# geometry constants (kept from your script)
Nd = 0.005
L = 2.0   # axial length (same as in your script)

# containers
eta_grid, psi_grid = np.meshgrid(eta_range, psi_range, indexing='xy')
x_array = np.zeros_like(eta_grid)
y_array = np.zeros_like(eta_grid)
z_pos_array = np.zeros_like(eta_grid)
z_neg_array = np.zeros_like(eta_grid)

# build raw shape arrays (same formulas as your Fig.26 reproduction)
for i_eta, eta in enumerate(eta_range):
    for j_psi, psi in enumerate(psi_range):

        Nc = cubic(psi)

        W = 1.0
        H = 0.5

        Sc = 0.5**(2 * Nc)             # (28)
        Cc = eta**Nc * (1 - eta)**Nc   # (29)

        Sd = 0.5**(2 * Nd)             # (30)
        Cd = psi**Nd * (1 - psi)**Nd   # (31)

        x = psi * L                    # (32)
        y = -(Sd * Cd) * (1 - 2 * eta) * W / 2   # (33)
        z = Sd * Cd * (Sc * Cc) * H / 2         # (34)

        x_array[i_eta, j_psi] = x
        y_array[i_eta, j_psi] = y
        z_pos_array[i_eta, j_psi] = z
        z_neg_array[i_eta, j_psi] = -z

# Normalise lateral and vertical lobes (keeps same visual proportions as original)
# Avoid zero-division
max_y = np.max(np.abs(y_array)) if np.max(np.abs(y_array)) > 0 else 1.0
max_zpos_cols = np.max(z_pos_array, axis=0)
max_zpos_cols[max_zpos_cols == 0] = 1.0

y_norm_array = y_array / max_y / 2.0  # roughly scaled to half-width
# Normalize each column of z_pos to avoid columnwise collapse
z_norm_pos_array = np.zeros_like(z_pos_array)
z_norm_neg_array = np.zeros_like(z_neg_array)
for j in range(psi_count):
    col_max = max_zpos_cols[j]
    z_norm_pos_array[:, j] = z_pos_array[:, j] / col_max / 2.0
    z_norm_neg_array[:, j] = z_neg_array[:, j] / (np.min(z_neg_array[:, j]) if np.min(z_neg_array[:, j]) != 0 else -1.0) / 2.0

# ---------- centreline (curved) ----------
# define centreline z_c(x) as cubic ramp from inlet_z to outlet_z
def centreline_z(x):
    """x in [0, L]"""
    t = np.clip(x / L, 0.0, 1.0)
    s = smooth_cubic(t) if use_smooth_centerline else t
    return inlet_z + (outlet_z - inlet_z) * s

# ---------- tilt angle alpha(x) ----------
def tilt_angle_rad(x):
    """tilt angle in radians at axial station x"""
    t = np.clip(x / L, 0.0, 1.0)
    s = smooth_cubic(t) if use_smooth_tilt else t
    return np.deg2rad(tilt_end_deg * s)

# ---------- apply rotation about local y and place cross-sections on curved centreline ----------
# We'll compute final 3D coordinates arrays for upper and lower lobes
rot_X_pos = np.zeros_like(x_array)   # X coordinates of rotated upper lobe
rot_Y_pos = np.zeros_like(y_norm_array)
rot_Z_pos = np.zeros_like(z_norm_pos_array)

rot_X_neg = np.zeros_like(x_array)   # X coords for lower lobe
rot_Y_neg = np.zeros_like(y_norm_array)
rot_Z_neg = np.zeros_like(z_norm_neg_array)

# Loop through grid and compute rotated+translated coordinates
for i in range(eta_count):
    for j in range(psi_count):
        base_x = x_array[i, j]              # axial station (pre-rotation)
        rel_y = y_norm_array[i, j]          # lateral offset relative to centreline
        rel_z_pos = z_norm_pos_array[i, j]  # positive lobe relative z (>=0)
        rel_z_neg = z_norm_neg_array[i, j]  # negative lobe relative z (<=0)

        # centreline coords at this axial station
        zc = centreline_z(base_x)
        yc = y_center

        # tilt at this station
        alpha = tilt_angle_rad(base_x)
        ca = np.cos(alpha)
        sa = np.sin(alpha)

        # rotate positive lobe relative vector [0, rel_y, rel_z_pos] about local y
        dx_pos = sa * rel_z_pos          # delta x introduced by tilt (sin* z)
        zpos_r = ca * rel_z_pos          # rotated local z (about y)
        x_pos_final = base_x + dx_pos
        y_pos_final = yc + rel_y
        z_pos_final = zc + zpos_r

        rot_X_pos[i, j] = x_pos_final
        rot_Y_pos[i, j] = y_pos_final
        rot_Z_pos[i, j] = z_pos_final

        # rotate negative lobe relative vector [0, rel_y, rel_z_neg] about local y
        dx_neg = sa * rel_z_neg
        zneg_r = ca * rel_z_neg
        x_neg_final = base_x + dx_neg
        y_neg_final = yc + rel_y
        z_neg_final = zc + zneg_r

        rot_X_neg[i, j] = x_neg_final
        rot_Y_neg[i, j] = y_neg_final
        rot_Z_neg[i, j] = z_neg_final

# ---------- Plot the lofted geometry ----------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# upper surface
ax.plot_surface(
    rot_X_pos, rot_Y_pos, rot_Z_pos,
    rstride=3, cstride=3, linewidth=0.3, edgecolor='gray',
    facecolor='lightcyan', alpha=0.85
)

# lower surface
ax.plot_surface(
    rot_X_neg, rot_Y_neg, rot_Z_neg,
    rstride=3, cstride=3, linewidth=0.3, edgecolor='gray',
    facecolor='lightcyan', alpha=0.85
)

# draw the centreline for reference
xc = np.linspace(0, L, 200)
zc_line = centreline_z(xc)
yc_line = np.ones_like(xc) * y_center
ax.plot(xc, yc_line, zc_line, color='k', linewidth=2, label='centreline')

ax.set_xlabel('X (axial)')
ax.set_ylabel('Y (lateral)')
ax.set_zlabel('Z (vertical)')
ax.set_title('3D Hollow Body: cross-section tilt + curved centreline')

ax.view_init(elev=20, azim=120)
# Keep aspect ratio visually balanced
try:
    ax.set_box_aspect((np.ptp(rot_X_pos), np.ptp(rot_Y_pos), np.ptp(np.concatenate((rot_Z_pos.ravel(), rot_Z_neg.ravel())))))
except Exception:
    pass

plt.legend()
plt.tight_layout()
plt.show()
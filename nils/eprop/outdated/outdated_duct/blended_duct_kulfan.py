import numpy as np
import matplotlib.pyplot as plt

# #%% Reproduce Fig. 21

# w = 2
# h = 2

# N_u = 0.5
# N_l = 0.25

# def calc_coordinates(N_u, N_l):
    
#     eta_range = np.linspace(0, 1, 1000)
    
#     y_u_list = []
#     y_l_list = []
#     z_u_list = []
#     z_l_list = []
    
#     for eta in eta_range:
        
#         y = eta - 0.5
        
#         zeta_u = 2 * eta**N_u * (1 - eta)**N_u
#         zeta_l = 2 * eta**N_l * (1 - eta)**N_l
        
#         zeta_u_max = 2 * (0.5**N_u) * (0.5**N_u)  # positive peak for upper lobe
#         zeta_l_max = 2 * (0.5**N_l) * (0.5**N_l)  # magnitude for lower lobe
        
#         zeta_u_norm = zeta_u / zeta_u_max / 2
#         zeta_l_norm = -zeta_l / zeta_l_max / 2
        
#         y_u_list.append(y)
#         y_l_list.append(y)
#         z_u_list.append(zeta_u_norm)
#         z_l_list.append(zeta_l_norm)
        
#     return y_u_list, y_l_list, z_u_list, z_l_list
    

# combinations = np.array([
#     [[0.5, 0.5], [0.25, 0.25], [0.005, 0.005]],
#     [[0.5, 0.25], [0.5, 0.05], [0.5, 0.005]],
#     [[1, 0.5], [2, 0.5], [5, 0.5]],
# ])

# # Create a 3x3 grid of subplots
# fig, axes = plt.subplots(3, 3, figsize=(9, 9), constrained_layout=True)

# for i, row in enumerate(combinations):
    
#     for j, column in enumerate(row):
        
#         ax = axes[i][j]
        
#         N_u, N_l = column
        
#         y_u_list, y_l_list, z_u_list, z_l_list = calc_coordinates(N_u, N_l)
        
#         ax.plot(y_u_list, z_u_list)
#         ax.plot(y_l_list, z_l_list)
        
#         ax.set_aspect('equal', 'box')
            
# plt.show()

#%% Reproduce Fig. 26

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
    
    
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(
    x_array, y_norm_array, z_norm_pos_array,
    # x_array, y_array, z_pos_array,
    rstride=4, cstride=4,         # stride controls how dense the grid lines are
    edgecolor='gray',             # mesh color
    facecolor='lightblue',        # fill color
    alpha=0.8                     # slightly transparent
)

ax.plot_surface(
    x_array, y_norm_array, z_norm_neg_array,
    # x_array, y_array, z_neg_array,
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
plt.tight_layout()
plt.show()



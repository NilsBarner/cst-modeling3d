import sys
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["get_guide_curve"]


def get_guide_curve(
    N_u, N_l,
    w, h,
    N=100,
):
    
    # From C:\Users\nmb48\Documents\GitHub\RCAIDE_LEADS\RCAIDE\Library\Methods\Utilities\Chebyshev\chebyshev_data.py
    # cosine spaced in range [0,1]
    eta_range = 0.5*(1 - np.cos(np.pi*np.arange(0,N)/(N-1)))
    
    y = (eta_range - 0.5) * w
    
    zeta_u = 2 * eta_range**N_u * (1 - eta_range)**N_u
    zeta_l = 2 * eta_range**N_l * (1 - eta_range)**N_l
    
    zeta_u_max = 2 * (0.5**N_u) * (0.5**N_u)
    zeta_l_max = 2 * (0.5**N_l) * (0.5**N_l)
    
    zeta_u_norm = zeta_u / zeta_u_max / 2
    zeta_l_norm = -zeta_l / zeta_l_max / 2
    
    z_u = zeta_u_norm * h
    z_l = zeta_l_norm * h
    
    r_u = np.sqrt(np.array(y)**2 + np.array(z_u)**2)
    theta_u = np.rad2deg(np.atan2(np.array(z_u), np.array(y)))
    
    r_l = np.sqrt(np.array(y)**2 + np.array(z_l)**2)
    theta_l = np.rad2deg(np.atan2(np.array(z_l), np.array(y)))
    
    return (
        y,
        z_u, z_l,
        r_u, r_l,
        theta_u, theta_l,
    )

#%%

if __name__ == '__main__':
    
    N_u = 0.1
    N_l = 0.1
    w = 2
    h = 1

    (
    y,
    z_u, z_l,
    r_u, r_l,
    theta_u, theta_l,
    ) = get_guide_curve(
        N_u, N_l,
        w, h,
    )
    
    
    fig, ax = plt.subplots()
    ax.plot(y, z_u, marker='.')
    ax.plot(y, z_l, marker='.')
    ax.set_aspect('equal')
    plt.show()
    
    
    fig, ax = plt.subplots()
    ax.plot(theta_u, r_u/np.min(r_u), marker='.')
    plt.show()



#!/usr/bin/env python3
"""
Minimal plot of spinner cross-section (uses Eq. C.1 and C.5 from Hawkswell thesis).
Produces a 2D line plot r(x) for x in [0,c].

Reference: Hawkswell Appendix C, Eq. C.1 and Eq. C.5. :contentReference[oaicite:1]{index=1}
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- spinner parameters (from thesis table / text)
c_mm = 212.0          # spinner chord in mm (thesis value)
c = c_mm / 1000.0     # convert to meters for plotting (optional)

# ---- S_hat polynomial (Eq. C.5)
def S_hat(psih):
    return -0.8408 * psih**3 + 1.2136 * psih**2 - 0.1941 * psih + 0.4

zetate = S_hat(1)

# ---- compute nondimensional radius using shape-space mapping (Eq. C.1 rearranged)
psih = np.linspace(0.0, 1.0, 501)            # nondim axial coordinate psi_hat = x / c
Svals = S_hat(psih)
zeta_hat = Svals * np.sqrt(psih) * (1 - psih) + psih * zetate

r = zeta_hat * c                             # physical radius in meters
x = psih * c                                  # physical axial coordinate in meters

# ---- optional: convert to mm for a "paper-style" plot
x_mm = x * 1000.0
r_mm = r * 1000.0

# ---- plot
plt.figure(figsize=(6,3.5))
plt.plot(x_mm, r_mm, '-', lw=2, color='k')
plt.fill_between(x_mm, r_mm, 0.0, color='lightgray', alpha=0.8)  # filled spinner cross-section
plt.grid(True, linestyle='--', alpha=0.4)
plt.xlabel('Axial x (mm)')
plt.ylabel('Radius r (mm)')
plt.title('Spinner cross-section')
plt.axis([0, c_mm, 0, max(r_mm)*1.05])
plt.gca().set_aspect('auto')
plt.tight_layout()
plt.show()

#%%

# plt.figure(figsize=(6,3.5))
# plt.plot(psih, Svals, '-', lw=2, color='k')
# plt.show()
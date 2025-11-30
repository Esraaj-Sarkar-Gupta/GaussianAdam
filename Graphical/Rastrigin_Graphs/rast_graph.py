#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 21:42:50 2025

@author: esraaj
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def rastrigin(x, y, A=10):
    """
    Computes the Rastrigin function for 2 dimensions.
    Global minimum is at (0,0) with value 0.
    Standard domain is usually [-5.12, 5.12].
    """
    n = 2
    return A * n + (x**2 - A * np.cos(2 * np.pi * x)) + \
           (y**2 - A * np.cos(2 * np.pi * y))

# 1. Setup the grid
x = np.linspace(-5.12, 5.12, 150)
y = np.linspace(-5.12, 5.12, 150)
X, Y = np.meshgrid(x, y)
Z = rastrigin(X, Y)

# 2. Create the plot
fig = plt.figure(figsize=(28, 12))

# --- Plot 1: 3D Surface ---
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf = ax1.plot_surface(X, Y, Z, cmap='viridis', 
                        edgecolor='none', alpha=0.9, antialiased=True)

# Add a contour projection on the floor of the 3D plot
ax1.contour(X, Y, Z, zdir='z', offset=-5, cmap='viridis', alpha=0.5)

ax1.set_title("Rastrigin Function (3D Surface)", fontsize=14)
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")
ax1.set_zlim(-5, np.max(Z)) # Set limit to see the floor projection
ax1.view_init(elev=45, azim=45) # Good angle to see the 'egg crate' shape

# --- Plot 2: 2D Contour (Heatmap) ---
ax2 = fig.add_subplot(1, 2, 2)
contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
fig.colorbar(contour, ax=ax2, label='Loss Value')

# Mark the Global Minimum
ax2.plot(0, 0, 'rx', markersize=10, markeredgewidth=2, label='Global Min (0,0)')
ax2.legend()

ax2.set_title("Rastrigin Function (2D Contour)", fontsize=14)
ax2.set_xlabel("X")
ax2.set_ylabel("Y")

plt.tight_layout()
plt.show()
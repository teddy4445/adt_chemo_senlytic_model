import numpy as np
import matplotlib.pyplot as plt

# Updated parameters for the simulation model

# Diffusion Coefficients
delta_C = 8.64e-3  # cm^2/d for Cancer Cells (n36)
delta_W = 0.8      # cm^2/d for Oxygen (n37)
delta_I = 6.05e-2  # cm^2/d for I_12 (n38)
delta_V = 8.64e-2  # cm^2/d for VEGF (n39)

# Death Rates
d_C = 0.1          # d^-1 for Cancer Cells (n37)
d_C_s = 0.92       # d^-1 for Senescent Cancer Cells (n41)
d_D = 0.1          # d^-1 for Dendritic Cells (n42)
d_T = 0.18         # d^-1 for T Cells (n42)
d_E = 0.69         # d^-1 for Endothelial Cells (n43)
d_W = 1.04         # d^-1 for Oxygen Uptake (n37)
d_I = 1.38         # d^-1 for I_12 Degradation (n41)
d_V = 12.6         # d^-1 for VEGF Degradation (n42)

# Carrying Capacities
C_0 = 0.8          # g/cm^3 for Cancer Cells (n42)
E_0 = 5e-3         # g/cm^3 for Endothelial Cells (n41)
D_0 = 2e-5         # g/cm^3 for Immature Dendritic Cells (n41)
T_0 = 2e-4         # g/cm^3 for Naive T Cells (n41)
W_0 = 4.65e-4      # g/cm^3 for Normal Oxygen Density (n43)
W_star = 1.69e-4   # g/cm^3 for Hypoxia Threshold (n43)
V_0 = 3.65e-10     # g/cm^3 for VEGF Threshold (n42)

# Saturation Constants
K_C = 0.4          # g/cm^3 for Cancer Cells (n38)
K_D = 4e-4         # g/cm^3 for Dendritic Cells (n37)
K_T = 1e-3         # g/cm^3 for T Cells (n37)
K_E = 2.5e-3       # g/cm^3 for Endothelial Cells (n42)
K_W = 1.69e-4      # g/cm^3 for Oxygen (n42)
K_I = 8e-10        # g/cm^3 for I_12 (n45)
K_V = 7e-8         # g/cm^3 for VEGF (n42)

# Growth and Production Rates
lambda_CW = 1.49   # d^-1 for Growth of Cancer Cells (Estimated)
lambda_CC_s = 0.092 # d^-1 for Production of Senescent Cancer Cells (Estimated)
lambda_D = 1.12    # d^-1 for Production of Dendritic Cells (Estimated)
lambda_T = 1.47    # d^-1 for Production of T Cells (Estimated)
lambda_EV = 1.87e7  # d^-1 for Production of Endothelial Cells (Estimated)
lambda_WE = 9.45e-2 # d^-1 for Oxygen Production by Endothelial Cells (Estimated)
lambda_ID = 5.52e-6 # d^-1 for Production of I_12 (Estimated)
lambda_VW = 2.44e-7 # d^-1 for VEGF Production by Oxygen (Estimated)

# Other Parameters
mu_TC = 500        # cm^3/g.d for Killing Rate of Cancer by T Cells (This Work)
d_TI = 2.76        # d^-1 for Loss of I_12 by T Cells (This Work)
d_EV = 25.2        # d^-1 for Loss of VEGF by Endothelial Cells (This Work)

# Other Rates
alpha = 5.32       # d^-1 for Decrease Rate of Fisetin (n32, n33)
beta = 0.174       # d^-1 for Decrease Rate of Cabazitaxel (n35)
mu_F = 2           # d^-1 for Washout Rate of Fisetin (This Work)
mu_P = 2           # d^-1 for Washout Rate of Cabazitaxel (This Work)

# Dose Amounts
gamma_F = 6.4e-4   # g/cm^3.d for Fisetin Dose (n17)
gamma_P = 6.4e-4   # g/cm^3.d for Cabazitaxel Dose (n17)


# Grid parameters
Nx = 100  # number of spatial grid points
Ny = 100
Nt = 1000  # number of time steps
dx = 1.0  # spatial step size
dy = 1.0
dt = 0.01  # time step size

# Create spatial grids for C, C_r, C_s, D, T, E, W, I, V, F, and P
C = np.zeros((Nx, Ny))
C_r = np.zeros((Nx, Ny))
C_s = np.zeros((Nx, Ny))
D = np.zeros((Nx, Ny))
T = np.zeros((Nx, Ny))
E = np.zeros((Nx, Ny))
W = np.ones((Nx, Ny))  # oxygen levels initialized
I = np.zeros((Nx, Ny))
V = np.zeros((Nx, Ny))
F = np.zeros((Nx, Ny))
P = np.zeros((Nx, Ny))

# Function to compute the logistic growth term for cancer cells
def lambda_W_func(W):
    return lambda_CW * np.where(W <= W_0, W / W_0, 1.0)

# Spatial derivative calculation (2nd order central difference)
def laplacian(field, dx, dy):
    return (np.roll(field, 1, axis=0) - 2*field + np.roll(field, -1, axis=0)) / dx**2 + \
           (np.roll(field, 1, axis=1) - 2*field + np.roll(field, -1, axis=1)) / dy**2

# Main time-stepping loop
for t in range(Nt):
    # Compute all the spatial and logistic terms for each variable
    lambda_W_val = lambda_W_func(W)
    
    # Eq. (1) for C (cancer cells)
    dC_dt = lambda_W_val * C * (1 - C / C_0) - mu_TC * T * C - mu_AC * A * C - mu_PC * P * C - d_C * C
    dC_dt += laplacian(C, dx, dy) - np.roll(C, 1, axis=0) * np.roll(C, 1, axis=1)  # Adjust for drift
    
    # Eq. (2) for C_r (castration-resistant cells)
    dC_r_dt = lambda_W_val * C_r * (1 - C_r / C_0) + lambda_CC_r * A * C - mu_TC * T * C_r - mu_PC * P * C_r - d_C * C_r
    dC_r_dt += laplacian(C_r, dx, dy) - np.roll(C_r, 1, axis=0) * np.roll(C_r, 1, axis=1)
    
    # Eq. (3) for C_s (senescent cells)
    dC_s_dt = lambda_CC_r * C - mu_F * F * C_s
    dC_s_dt += laplacian(C_s, dx, dy) - np.roll(C_s, 1, axis=0) * np.roll(C_s, 1, axis=1)
    
    # Eq. (4) for D (dendritic cells)
    dD_dt = lambda_D * D * C / (K_C + C) - d_D * D
    dD_dt += laplacian(D, dx, dy) - np.roll(D, 1, axis=0) * np.roll(D, 1, axis=1)
    
    # Eq. (5) for T (T cells)
    dT_dt = lambda_T * T * I / (K_I + I) - mu_PT * T * P - d_T * T
    dT_dt += laplacian(T, dx, dy) - np.roll(T, 1, axis=0) * np.roll(T, 1, axis=1)
    
    # Eq. (6) for E (endothelial cells)
    dE_dt = lambda_E * E * (1 - E / E_0) - d_E * E
    dE_dt += laplacian(E, dx, dy) - np.roll(E, 1, axis=0) * np.roll(E, 1, axis=1)
    
    # Eq. (7) for W (oxygen level)
    dW_dt = lambda_W * E - d_W * W
    dW_dt += laplacian(W, dx, dy)
    
    # Eq. (8) for V (VEGF)
    dV_dt = lambda_VW * (C + C_r + C_s) - d_V * V
    dV_dt += laplacian(V, dx, dy) - np.roll(V, 1, axis=0) * np.roll(V, 1, axis=1)
    
    # Eq. (9) for I
    dI_dt = lambda_ID * D - d_T * I * T / (K_T + T) - d_I * I
    dI_dt += laplacian(I, dx, dy)
    
    # Eq. (10) for F
    dF_dt = gamma_F * f_F_alpha(t) - mu_F * F - mu_C_s_F * F
    dF_dt += laplacian(F, dx, dy)
    
    # Eq. (11) for P (chemotherapy)
    dP_dt = gamma_P * f_P_beta(t) - mu_CP * C * P - mu_C_P * C_r * P - mu_TP * T * P - mu_P * P
    dP_dt += laplacian(P, dx, dy)
    
    # Update the variables using explicit time-stepping
    C += dC_dt * dt
    C_r += dC_r_dt * dt
    C_s += dC_s_dt * dt
    D += dD_dt * dt
    T += dT_dt * dt
    E += dE_dt * dt
    W += dW_dt * dt
    V += dV_dt * dt
    I += dI_dt * dt
    F += dF_dt * dt
    P += dP_dt * dt
    
    # Optional: Plot or save the data every certain time steps
    if t % 100 == 0:
        plt.imshow(C, cmap='hot')
        plt.colorbar()
        plt.title(f"Time step {t}")
        plt.show()

# Example of plotting the final state of cancer cells
plt.imshow(C, cmap='hot')
plt.colorbar()
plt.show()
# DGNS-Engine-V5: Ducks Guts Navier Stokes Tensor Engine (Version 5)
# Purpose: Simulate compressible Navier-Stokes equations in 3D to map negative uncomputable space
# Dependencies: numpy, torch, matplotlib, plotly, imageio (for animation), argparse, os, time
# Assumptions:
# - Ideal gas law for air at STP (T = 273.15 K, P = 101325 Pa, gamma = 1.4, R = 287 J/kg·K)
# - Lid-driven cavity setup with user-configurable initial conditions
# - Uses Smagorinsky LES model for subgrid-scale turbulence
# - Spectral smoothing to stabilize high-frequency modes
# - StabilityNet neural network for adaptive time-stepping
# Version History:
# - V5 (May 28, 2025): Introduced compressible flow, ideal gas law, user-configurable gas properties
# Authors: Aetheris Navigatrix & Grok 3 (xAI)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import time
import os
import argparse
import imageio

# Set default renderer for plotly
pio.renderers.default = "browser"

# Set device to CPU
device = torch.device("cpu")
print(f"Using device: {device}")

# Parse command-line arguments for simulation parameters
parser = argparse.ArgumentParser(description="DGNS-Engine-V5: Compressible Navier-Stokes Simulation")
parser.add_argument('--N', type=int, default=32, help='Grid size (default: 32)')  # Increased resolution
parser.add_argument('--steps', type=int, default=200, help='Number of steps (default: 200)')
parser.add_argument('--reynolds', type=float, default=50000, help='Reynolds number (default: 50000)')
parser.add_argument('--dt', type=float, default=0.0001, help='Time step (default: 0.0001)')
parser.add_argument('--T0', type=float, default=273.15, help='Initial temperature in K (default: 273.15 K, STP)')
parser.add_argument('--P0', type=float, default=101325, help='Initial pressure in Pa (default: 101325 Pa, STP)')
parser.add_argument('--gamma', type=float, default=1.4, help='Adiabatic index for air (default: 1.4)')
parser.add_argument('--R', type=float, default=287.0, help='Gas constant for air (default: 287 J/kg·K)')
parser.add_argument('--Cs', type=float, default=0.01, help='Smagorinsky constant (default: 0.01)')
args = parser.parse_args()

# Simulation parameters setup
N = args.N
dx = 1.0 / (N - 1)
dt = args.dt * (16 / N)**2  # Scale dt with grid size
steps = args.steps
nu = 1.0 / args.reynolds
dt_min = 1e-8
dx_min = 0.002
divergence_threshold = 1e-6
Cs = args.Cs
max_iterations = 1
R = args.R
gamma = args.gamma
cv = R / (gamma - 1)
T0 = args.T0
P0 = args.P0
rho0 = P0 / (R * T0)

# Create output directory for checkpoints
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

# Initialize state tensor: [rho, rho*u, rho*v, rho*w, rho*E]
S = torch.zeros((N, N, N, 5), dtype=torch.float32, device=device)
S[:, :, :, 0] = rho0
S[:, :, -1, 1] = rho0 * 5.0  # Reduced lid velocity to 5 m/s
torch.manual_seed(42)
S[:, :, :, 1:4] += 0.05 * rho0 * torch.randn(N, N, N, 3, device=device)  # Further reduced perturbation
# Add a density perturbation in the center
center = N // 2
S[center-2:center+2, center-2:center+2, center-2:center+2, 0] *= 1.2  # 20% density increase
u = S[:, :, :, 1] / S[:, :, :, 0]
v = S[:, :, :, 2] / S[:, :, :, 0]
w = S[:, :, :, 3] / S[:, :, :, 0]
S[:, :, :, 4] = S[:, :, :, 0] * (cv * T0 + 0.5 * (u**2 + v**2 + w**2))
S.requires_grad = True

# Metadata tensor for uncomputable regions
M = torch.zeros((N, N, N), dtype=torch.int32, device=device)

# Log initial stability metrics
def log_initial_conditions(S, dx):
    grads = compute_gradients(S, dx)
    grad_norm = torch.sqrt(sum(g**2 for g in grads)).mean()
    rho = S[:, :, :, 0].unsqueeze(-1)
    velocities = S[:, :, :, 1:4] / rho
    variance = torch.var(velocities)
    print(f"Initial Conditions: Grad Norm: {grad_norm:.2e}, Variance: {variance:.2e}")

log_initial_conditions(S, dx)

# Stability network to adjust time step dynamically
class StabilityNet(nn.Module):
    def __init__(self):
        super(StabilityNet, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

stability_net = StabilityNet().to(device)
optimizer = optim.Adam(stability_net.parameters(), lr=0.001)
criterion = nn.MSELoss()

training_data = []
training_labels = []

# Finite difference helpers for gradient and Laplacian calculations
def compute_gradients(S, dx):
    grad = lambda x: (x[2:, 1:-1, 1:-1] - x[:-2, 1:-1, 1:-1]) / (2 * dx)
    grad_y = lambda x: (x[1:-1, 2:, 1:-1] - x[1:-1, :-2, 1:-1]) / (2 * dx)
    grad_z = lambda x: (x[1:-1, 1:-1, 2:] - x[1:-1, 1:-1, :-2]) / (2 * dx)
    
    grads = []
    for i in range(5):
        grads.extend([grad(S[:, :, :, i]), grad_y(S[:, :, :, i]), grad_z(S[:, :, :, i])])
    return grads

def laplacian(S, idx, dx):
    lap = (S[2:, 1:-1, 1:-1, idx] + S[:-2, 1:-1, 1:-1, idx] +
           S[1:-1, 2:, 1:-1, idx] + S[1:-1, :-2, 1:-1, idx] +
           S[1:-1, 1:-1, 2:, idx] + S[1:-1, 1:-1, :-2, idx] -
           6 * S[1:-1, 1:-1, 1:-1, idx]) / (dx**2)
    return lap

def divergence(S, dx):
    rho = S[:, :, :, 0]
    rho_u = S[:, :, :, 1]
    rho_v = S[:, :, :, 2]
    rho_w = S[:, :, :, 3]
    grad = lambda x: (x[2:, 1:-1, 1:-1] - x[:-2, 1:-1, 1:-1]) / (2 * dx)
    grad_y = lambda x: (x[1:-1, 2:, 1:-1] - x[1:-1, :-2, 1:-1]) / (2 * dx)
    grad_z = lambda x: (x[1:-1, 1:-1, 2:] - x[1:-1, 1:-1, :-2]) / (2 * dx)
    div = grad(rho_u) + grad_y(rho_v) + grad_z(rho_w)
    div_full = torch.zeros_like(rho)
    div_full[1:-1, 1:-1, 1:-1] = div
    return div_full

# LES model for subgrid-scale turbulence
def compute_sgs_term(S, dx):
    rho = S[:, :, :, 0]
    u = S[:, :, :, 1] / rho
    v = S[:, :, :, 2] / rho
    w = S[:, :, :, 3] / rho
    S_vel = torch.stack([u, v, w], dim=-1)
    grads = compute_gradients(torch.stack([u, v, w], dim=-1), dx)
    Sxx, Sxy, Sxz = grads[0], 0.5 * (grads[1] + grads[3]), 0.5 * (grads[2] + grads[6])
    Syy, Syz = grads[4], 0.5 * (grads[5] + grads[7])
    Szz = grads[8]
    S_mag = torch.sqrt(2 * (Sxx**2 + Syy**2 + Szz**2 + 2 * (Sxy**2 + Syz**2 + Sxz**2)))
    nu_t = (Cs * dx)**2 * S_mag
    nu_t_full = torch.zeros_like(rho)[1:-1, 1:-1, 1:-1]
    nu_t_full[:] = nu_t
    return nu_t_full

# Spectral smoothing to stabilize high-frequency modes
def spectral_smooth(S, N, dx):
    for i in range(5):
        var = S[:, :, :, i]
        var_hat = torch.fft.fftn(var)
        k = torch.fft.fftfreq(N, d=dx, device=device) * 2 * np.pi
        kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
        k2 = kx**2 + ky**2 + kz**2
        filter_mask = (k2 < (N * np.pi / 2)**2).float()
        var_hat_filtered = var_hat * filter_mask
        var_hat_filtered *= 0.9
        S[:, :, :, i] = torch.fft.ifftn(var_hat_filtered).real
    return S

# Check stability and flag uncomputable regions
def check_stability_and_computability(S, t, M, dx):
    grads = compute_gradients(S, dx)
    grad_norm = torch.sqrt(sum(g**2 for g in grads)).mean()
    rho = S[:, :, :, 0].unsqueeze(-1)
    velocities = S[:, :, :, 1:4] / rho
    variance = torch.var(velocities)
    
    # Log stability metrics
    print(f"Stability Check (Step {t:.4f}): Grad Norm: {grad_norm:.2e}, Variance: {variance:.2e}")
    
    # Flag regions pointwise with a higher threshold
    adjusted_dt = dt
    grad_norm_tensor = torch.zeros_like(S[:, :, :, 0])
    for i, grad in enumerate(grads):
        grad_norm_tensor[1:-1, 1:-1, 1:-1] += grad**2
    grad_norm_tensor = torch.sqrt(grad_norm_tensor)
    uncomputable_mask = (grad_norm_tensor > 5e3) | torch.isnan(grad_norm_tensor) | torch.isinf(grad_norm_tensor)  # Increased threshold
    M[uncomputable_mask] = 1
    
    stability_input = torch.tensor([[grad_norm.item(), variance.item(), t]], dtype=torch.float32, device=device)
    adjustment = stability_net(stability_input)
    adjusted_dt = adjustment.item() * dt
    print(f"Adjusted dt: {adjusted_dt:.2e}")
    if adjusted_dt < dt_min:
        print(f"Flagging additional uncomputable regions at step {t:.4f} due to small dt: {adjusted_dt:.2e}")
        M[1:-1, 1:-1, 1:-1] = 1
        adjusted_dt = dt_min

    training_data.append([grad_norm.item(), variance.item(), t])
    training_labels.append(1.0 if adjusted_dt > dt_min else 0.0)
    return adjusted_dt, M, grad_norm

# Compressible Navier-Stokes time step
def navier_stokes_step(S, dt, nu, R, gamma, cv, M, N, dx):
    # Compute fraction of uncomputable regions
    uncomputable_fraction = M[1:-1, 1:-1, 1:-1].sum().item() / ((N-2)**3)
    if uncomputable_fraction > 0.999:
        print(f"Stopping Navier-Stokes step: {uncomputable_fraction:.2%} of domain is uncomputable")
        return S, M

    rho = S[:, :, :, 0]
    rho_u = S[:, :, :, 1]
    rho_v = S[:, :, :, 2]
    rho_w = S[:, :, :, 3]
    rho_E = S[:, :, :, 4]
    u = rho_u / rho
    v = rho_v / rho
    w = rho_w / rho
    E = rho_E / rho
    vel2 = u**2 + v**2 + w**2
    T = (E - 0.5 * vel2) / cv
    P = rho * R * T
    grads = compute_gradients(S, dx)

    # Log max values for debugging
    print(f"Navier-Stokes Step: Max rho: {rho.max().item():.2e}, Max u: {u.max().item():.2e}, Max P: {P.max().item():.2e}")

    # Continuity equation
    rho_flux_x = grads[0]
    rho_flux_y = grads[1]
    rho_flux_z = grads[2]
    rho_new = rho[1:-1, 1:-1, 1:-1] - dt * (rho_flux_x + rho_flux_y + rho_flux_z)

    # Momentum equations
    nu_t = compute_sgs_term(S, dx)
    effective_nu = nu + nu_t
    S_new = S.clone()
    for i, vel in enumerate([u, v, w]):
        idx = i + 1
        grad_vel_x, grad_vel_y, grad_vel_z = grads[3*i:3*(i+1)]
        grad_rho_vel_x, grad_rho_vel_y, grad_rho_vel_z = grads[3*i+3:3*(i+1)+3]
        grad_P_x, grad_P_y, grad_P_z = grads[12+3*i:12+3*(i+1)]
        lap_vel = laplacian(torch.stack([vel], dim=-1), 0, dx)
        conv = rho[1:-1, 1:-1, 1:-1] * (u[1:-1, 1:-1, 1:-1] * grad_vel_x + 
                                        v[1:-1, 1:-1, 1:-1] * grad_vel_y + 
                                        w[1:-1, 1:-1, 1:-1] * grad_vel_z)
        S_new[1:-1, 1:-1, 1:-1, idx] = (rho[1:-1, 1:-1, 1:-1] * vel[1:-1, 1:-1, 1:-1] - 
                                        dt * (conv + grad_P_x + grad_P_y + grad_P_z - 
                                              effective_nu * lap_vel))

    # Energy equation
    grad_rho_E_x, grad_rho_E_y, grad_rho_E_z = grads[12:15]
    div_Pu = P[1:-1, 1:-1, 1:-1] * (grads[3] + grads[6] + grads[9])
    rho_E_new = rho_E[1:-1, 1:-1, 1:-1] - dt * (grad_rho_E_x + grad_rho_E_y + grad_rho_E_z + div_Pu)

    # Update state (only update computable regions)
    S_new[1:-1, 1:-1, 1:-1, 0] = torch.where(M[1:-1, 1:-1, 1:-1] == 0, rho_new, S[1:-1, 1:-1, 1:-1, 0])
    S_new[1:-1, 1:-1, 1:-1, 4] = torch.where(M[1:-1, 1:-1, 1:-1] == 0, rho_E_new, S[1:-1, 1:-1, 1:-1, 4])
    S_new = spectral_smooth(S_new, N, dx)
    return S_new, M

# Main simulation loop
results = {}
divergence_history = []
energy_history = []
negative_space_history = []
time_per_step = []

print("Running DGNS-Engine-V5 simulation...")
start_time = time.time()
for t in range(steps):
    step_start = time.time()
    adjusted_dt, M, grad_norm = check_stability_and_computability(S, t * dt, M, dx)
    S, M = navier_stokes_step(S, adjusted_dt, nu, R, gamma, cv, M, N, dx)
    step_time = time.time() - step_start
    time_per_step.append(step_time)

    rho = S[:, :, :, 0]
    u = S[:, :, :, 1] / rho
    v = S[:, :, :, 2] / rho
    w = S[:, :, :, 3] / rho
    div = divergence(S, dx).abs().mean().item()
    energy = 0.5 * (rho * (u**2 + v**2 + w**2)).sum().item()
    negative_space_fraction = M.sum().item() / (N**3)
    divergence_history.append(div)
    energy_history.append(energy)
    negative_space_history.append(negative_space_fraction)

    if torch.any(torch.isnan(S)):
        print(f"Simulation crashed at step {t}")
        break

    if t % 10 == 0 or t == steps - 1:
        print(f"Step {t}/{steps}, Div: {div:.2e}, Energy: {energy:.2f}, "
              f"Negative Space: {negative_space_fraction:.2%}, Time: {step_time:.2f}s")
        torch.save({
            'step': t,
            'S': S,
            'M': M,
            'divergence_history': divergence_history,
            'energy_history': energy_history,
            'negative_space_history': negative_space_history
        }, f"checkpoints/checkpoint_step_{t}.pt")

# Train StabilityNet
if training_data:
    print("\nTraining StabilityNet...")
    X = torch.tensor(training_data, dtype=torch.float32, device=device)
    y = torch.tensor(training_labels, dtype=torch.float32, device=device).view(-1, 1)
    for epoch in range(5):
        optimizer.zero_grad()
        outputs = stability_net(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Visualizations
plt.figure(figsize=(8, 6))

plt.subplot(2, 2, 1)
plt.plot(divergence_history, label="Divergence")
plt.xlabel("Step")
plt.ylabel("Mean Abs Divergence")
plt.title("Divergence Over Time")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(energy_history, label="Energy")
plt.xlabel("Step")
plt.ylabel("Kinetic Energy")
plt.title("Energy Over Time")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(negative_space_history, label="Negative Space")
plt.xlabel("Step")
plt.ylabel("Fraction")
plt.title("Negative Uncomputable Space")
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
u_slice = (S[:, :, N//2, 1] / S[:, :, N//2, 0]).cpu().detach().numpy()
plt.imshow(u_slice, cmap="viridis", origin="lower", vmin=-2, vmax=5)
plt.colorbar(label="u-velocity")
plt.xlabel("x")
plt.ylabel("y")
plt.title("u-Velocity at Midplane")

plt.tight_layout()
plt.savefig("results.png")
plt.show()
plt.close()

# 3D Voxel Plot for negative space with focus on boundaries
M_np = M.cpu().detach().numpy()
x, y, z = np.indices(M_np.shape)
fig = go.Figure(data=go.Volume(
    x=x.flatten(), y=y.flatten(), z=z.flatten(),
    value=M_np.flatten(), isomin=0.5, isomax=1,
    opacity=0.2, surface_count=30, colorscale="Reds"  # Increased detail
))
fig.update_layout(
    title="Negative Uncomputable Space (M=1 Regions)",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z")
)
fig.write_html("negative_space.html")
fig.show()

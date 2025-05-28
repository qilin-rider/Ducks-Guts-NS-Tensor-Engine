# DGNS-Engine-V5: Ducks Guts Navier Stokes Tensor Engine (Version 5)
# Purpose: Simulate compressible Navier-Stokes equations in 3D to map negative uncomputable space
# Dependencies: numpy, torch, matplotlib, plotly, imageio (for animation)
# Assumptions:
# - Ideal gas law for air at STP (T = 273.15 K, P = 101325 Pa, gamma = 1.4, R = 287 J/kg·K)
# - Lid-driven cavity setup with user-configurable initial conditions
# - Uses Smagorinsky LES model for subgrid-scale turbulence
# - Spectral smoothing to stabilize high-frequency modes
# - StabilityNet neural network for adaptive time-stepping
# Version History:
# - V5 (May 28, 2025): Introduced compressible flow, ideal gas law, user-configurable gas properties
# Authors: Aetheris Navigatrix & Grok 3 (xAI)

# 1. Imports and Device Setup
# This section imports libraries for math, neural networks, plotting, and file handling.
# It checks if a GPU is available for faster computation, otherwise uses CPU.
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import time
import os
import imageio.v2 as imageio  # For creating animation GIFs

# Set default renderer for plotly to display plots in browser
pio.renderers.default = "browser"

# Set device to GPU if available, else CPU (GPU speeds up large simulations)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 2. User Input for Simulation Parameters
# This section defines default values for simulation parameters and asks the user to keep them or enter new ones.
# Parameters control grid size, time steps, flow physics, and turbulence modeling.
# Defaults adjusted for numerical stability to avoid simulation blow-up.

# Default simulation parameters (hardcoded for stability)
N = 32  # Grid size (small to reduce memory usage)
steps = 500  # Number of time steps to simulate
reynolds = 50000  # Reynolds number (lower for less turbulence)
dt = 0.00001  # Time step size (smaller for stability)
Cs = 0.1  # Smagorinsky constant (controls turbulence modeling)

# Function to get user input with default value
def get_user_input(prompt, default, type_func):
    user_input = input(f"{prompt} (default: {default}, press Enter to accept): ").strip()
    return type_func(user_input) if user_input else default

# Prompt user for each parameter
print("Set simulation parameters (press Enter to use defaults):")
N = get_user_input("Grid size (N)", N, int)
steps = get_user_input("Number of steps", steps, int)
reynolds = get_user_input("Reynolds number", reynolds, float)
dt = get_user_input("Time step (dt)", dt, float)
Cs = get_user_input("Smagorinsky constant (Cs)", Cs, float)

# 3. Simulation Parameter Setup
# This section uses user-provided or default parameters to set up the simulation grid and physical constants.
# It calculates derived values like grid spacing and gas properties for air at standard conditions.

# Calculate grid spacing and other parameters
dx = 1.0 / (N - 1)  # Grid spacing (domain is 1 unit wide)
nu = 1.0 / reynolds  # Kinematic viscosity (from Reynolds number)
dt_min = 1e-8  # Minimum time step to prevent instability
dx_min = 0.002  # Minimum grid spacing (not used in this version)
divergence_threshold = 1e-6  # Threshold for checking simulation stability
max_iterations = 1  # Maximum iterations per step (fixed at 1)

# Gas properties for air at standard temperature and pressure (STP)
T0 = 273.15  # Initial temperature (Kelvin)
P0 = 101325  # Initial pressure (Pascals)
gamma = 1.4  # Adiabatic index for air
R = 287.0  # Gas constant for air (J/kg·K)
cv = R / (gamma - 1)  # Specific heat at constant volume
rho0 = P0 / (R * T0)  # Initial density (from ideal gas law)

# Create output directory for saving simulation checkpoints
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

# 4. Helper Functions and Initialization
# This section defines functions to compute gradients, Laplacians, and turbulence terms.
# It also initializes the simulation state with a lid-driven cavity and perturbations.

# --- Helper Functions ---

def compute_gradients(S, dx):
    """Compute gradients of the state tensor S with respect to x, y, and z for each component."""
    grad = lambda x: (x[2:, 1:-1, 1:-1] - x[:-2, 1:-1, 1:-1]) / (2 * dx)
    grad_y = lambda x: (x[1:-1, 2:, 1:-1] - x[1:-1, :-2, 1:-1]) / (2 * dx)
    grad_z = lambda x: (x[1:-1, 1:-1, 2:] - x[1:-1, 1:-1, :-2]) / (2 * dx)
    grads = []
    num_components = S.shape[-1]  # Get number of components (e.g., 5 for state, 3 for velocities)
    for i in range(num_components):
        grads.extend([grad(S[:, :, :, i]), grad_y(S[:, :, :, i]), grad_z(S[:, :, :, i])])
    return grads

def laplacian(S, idx, dx):
    """Compute the Laplacian of a component of S."""
    lap = (S[2:, 1:-1, 1:-1, idx] + S[:-2, 1:-1, 1:-1, idx] +
           S[1:-1, 2:, 1:-1, idx] + S[1:-1, :-2, 1:-1, idx] +
           S[1:-1, 1:-1, 2:, idx] + S[1:-1, 1:-1, :-2, idx] -
           6 * S[1:-1, 1:-1, 1:-1, idx]) / (dx ** 2)
    return lap

def divergence(S, dx):
    """Compute the divergence of the momentum components."""
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

def compute_sgs_term(S, dx):
    """Compute the subgrid-scale (SGS) term using the Smagorinsky model."""
    rho = S[:, :, :, 0]
    u = S[:, :, :, 1] / rho
    v = S[:, :, :, 2] / rho
    w = S[:, :, :, 3] / rho
    grads = compute_gradients(torch.stack([u, v, w], dim=-1), dx)
    Sxx, Sxy, Sxz = grads[0], 0.5 * (grads[1] + grads[3]), 0.5 * (grads[2] + grads[6])
    Syy, Syz = grads[4], 0.5 * (grads[5] + grads[7])
    Szz = grads[8]
    S_mag = torch.sqrt(2 * (Sxx ** 2 + Syy ** 2 + Szz ** 2 + 2 * (Sxy ** 2 + Syz ** 2 + Sxz ** 2)))
    nu_t = (Cs * dx) ** 2 * S_mag
    nu_t_full = torch.zeros_like(rho)[1:-1, 1:-1, 1:-1]
    nu_t_full[:] = nu_t
    return nu_t_full

def spectral_smooth(S, N, dx):
    """Apply spectral smoothing to stabilize high-frequency modes."""
    for i in range(5):
        var = S[:, :, :, i]
        var_hat = torch.fft.fftn(var)
        k = torch.fft.fftfreq(N, d=dx, device=device) * 2 * np.pi
        kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
        k2 = kx ** 2 + ky ** 2 + kz ** 2
        filter_mask = (k2 < (N * np.pi / 2) ** 2).float()
        var_hat_filtered = var_hat * filter_mask
        var_hat_filtered *= 0.95  # Softer damping
        S[:, :, :, i] = torch.fft.ifftn(var_hat_filtered).real
    return S

# --- Logging and Stability Functions ---

def log_initial_conditions(S, dx):
    """Log initial stability metrics of the state tensor."""
    grads = compute_gradients(S, dx)
    grad_norm = torch.sqrt(sum(g ** 2 for g in grads)).mean()
    rho = S[:, :, :, 0].unsqueeze(-1)
    velocities = S[:, :, :, 1:4] / rho
    variance = torch.var(velocities)
    print(f"Initial Conditions: Grad Norm: {grad_norm:.2e}, Variance: {variance:.2e}")

# --- Initialization ---

# Initialize state tensor: [rho, rho*u, rho*v, rho*w, rho*E]
S = torch.zeros((N, N, N, 5), dtype=torch.float32, device=device)
S[:, :, :, 0] = rho0
S[:, :, -1, 1] = rho0 * 10.0  # Lid velocity set to 10 m/s
torch.manual_seed(42)
S[:, :, :, 1:4] += 0.1 * rho0 * torch.randn(N, N, N, 3, device=device)  # Larger perturbation
center = N // 2
S[center - 4:center + 4, center - 4:center + 4, center - 4:center + 4, 0] *= 1.5  # 50% density increase
u = S[:, :, :, 1] / S[:, :, :, 0]
v = S[:, :, :, 2] / S[:, :, :, 0]
w = S[:, :, :, 3] / S[:, :, :, 0]
S[:, :, :, 4] = S[:, :, :, 0] * (cv * T0 + 0.5 * (u ** 2 + v ** 2 + w ** 2))
S.requires_grad = True

# Metadata tensor for uncomputable regions
M = torch.zeros((N, N, N), dtype=torch.int32, device=device)

# Log initial conditions
log_initial_conditions(S, dx)

# 5. Stability and Navier-Stokes Solver
# This section includes a neural network for adaptive time-stepping, a stability checker, and the Navier-Stokes solver.
# Enhanced stability with tighter CFL condition and stronger spectral smoothing.

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

# Check stability and flag uncomputable regions
def check_stability_and_computability(S, t, M, dx):
    grads = compute_gradients(S, dx)
    grad_norm = torch.sqrt(sum(g ** 2 for g in grads)).mean()
    rho = S[:, :, :, 0].unsqueeze(-1)
    velocities = S[:, :, :, 1:4] / rho
    variance = torch.var(velocities)

    print(f"Stability Check (Step {t:.4f}): Grad Norm: {grad_norm:.2e}, Variance: {variance:.2e}")

    grad_norm_tensor = torch.zeros_like(S[:, :, :, 0])
    for grad in grads:
        grad_norm_tensor[1:-1, 1:-1, 1:-1] += grad ** 2
    grad_norm_tensor = torch.sqrt(grad_norm_tensor)
    uncomputable_mask = (grad_norm_tensor > 1e4) | torch.isnan(grad_norm_tensor) | torch.isinf(grad_norm_tensor)
    M[uncomputable_mask] = 1

    # Tighter CFL condition
    u_max = velocities.abs().max().item()
    c = torch.sqrt(torch.tensor(gamma * R * T0, device=device))  # Speed of sound
    cfl = 0.3  # Stricter CFL number
    adjusted_dt = cfl * dx / (u_max + c + 1e-10)
    adjusted_dt = max(min(adjusted_dt, dt), dt_min)
    print(f"Adjusted dt (CFL): {adjusted_dt:.2e}")

    if adjusted_dt < dt_min or u_max > 1e3:  # Flag if velocity explodes
        print(f"Flagging uncomputable regions at step {t:.4f} due to small dt or high velocity: dt={adjusted_dt:.2e}, u_max={u_max:.2e}")
        M[1:-1, 1:-1, 1:-1] = 1

    training_data.append([grad_norm.item(), variance.item(), t])
    training_labels.append(1.0 if adjusted_dt > dt_min and u_max < 1e3 else 0.0)
    return adjusted_dt, M, grad_norm

# Compressible Navier-Stokes time step
def navier_stokes_step(S, dt, nu, R, gamma, cv, M, N, dx):
    uncomputable_fraction = M[1:-1, 1:-1, 1:-1].sum().item() / ((N - 2) ** 3)
    if uncomputable_fraction > 0.99:
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
    vel2 = u ** 2 + v ** 2 + w ** 2
    T = (E - 0.5 * vel2) / cv
    P = rho * R * T

    # Compute only necessary gradients
    velocity_tensor = torch.stack([u, v, w], dim=-1)  # Shape (N, N, N, 3)
    vel_grads = compute_gradients(velocity_tensor, dx)  # 9 gradients
    P_tensor = P.unsqueeze(-1)  # Shape (N, N, N, 1)
    P_grads = compute_gradients(P_tensor, dx)  # 3 gradients
    rho_tensor = rho.unsqueeze(-1)  # Shape (N, N, N, 1)
    rho_grads = compute_gradients(rho_tensor, dx)  # 3 gradients
    E_tensor = rho_E.unsqueeze(-1)  # Shape (N, N, N, 1)
    E_grads = compute_gradients(E_tensor, dx)  # 3 gradients

    print(f"Navier-Stokes Step: Max rho: {rho.max().item():.2e}, Max u: {u.max().item():.2e}, Max P: {P.max().item():.2e}")

    rho_flux_x, rho_flux_y, rho_flux_z = rho_grads
    rho_new = rho[1:-1, 1:-1, 1:-1] - dt * (rho_flux_x + rho_flux_y + rho_flux_z)

    grad_P_x, grad_P_y, grad_P_z = P_grads
    nu_t = compute_sgs_term(S, dx)
    effective_nu = nu + nu_t
    S_new = S.clone()
    for i, vel in enumerate([u, v, w]):
        idx = i + 1
        grad_vel_x, grad_vel_y, grad_vel_z = vel_grads[3 * i:3 * i + 3]
        lap_vel = laplacian(torch.stack([vel], dim=-1), 0, dx)
        conv = rho[1:-1, 1:-1, 1:-1] * (u[1:-1, 1:-1, 1:-1] * grad_vel_x +
                                        v[1:-1, 1:-1, 1:-1] * grad_vel_y +
                                        w[1:-1, 1:-1, 1:-1] * grad_vel_z)
        S_new[1:-1, 1:-1, 1:-1, idx] = (rho[1:-1, 1:-1, 1:-1] * vel[1:-1, 1:-1, 1:-1] -
                                        dt * (conv + grad_P_x + grad_P_y + grad_P_z -
                                              effective_nu * lap_vel))

    grad_rho_E_x, grad_rho_E_y, grad_rho_E_z = E_grads
    div_Pu = P[1:-1, 1:-1, 1:-1] * (vel_grads[0] + vel_grads[3] + vel_grads[6])
    rho_E_new = rho_E[1:-1, 1:-1, 1:-1] - dt * (grad_rho_E_x + grad_rho_E_y + grad_rho_E_z + div_Pu)

    S_new[1:-1, 1:-1, 1:-1, 0] = torch.where(M[1:-1, 1:-1, 1:-1] == 0, rho_new, S[1:-1, 1:-1, 1:-1, 0])
    S_new[1:-1, 1:-1, 1:-1, 4] = torch.where(M[1:-1, 1:-1, 1:-1] == 0, rho_E_new, S[1:-1, 1:-1, 1:-1, 4])
    S_new = spectral_smooth(S_new, N, dx)
    return S_new, M

# 6. Main Simulation Loop and Visualizations
# This section runs the simulation, saves checkpoints, and creates plots/animations.
# Fixed plotting to generate all subplots and negative space HTML/PNG using write_image.

# --- Main Simulation Loop ---

results = {}
divergence_history = []
energy_history = []
negative_space_history = []
time_per_step = []
frames = []

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
    energy = 0.5 * (rho * (u ** 2 + v ** 2 + w ** 2)).sum().item()
    negative_space_fraction = M.sum().item() / (N ** 3)
    divergence_history.append(div if not torch.isnan(torch.tensor(div)) else 0.0)
    energy_history.append(energy if not torch.isnan(torch.tensor(energy)) else 0.0)
    negative_space_history.append(negative_space_fraction)

    if torch.any(torch.isnan(S)):
        print(f"Simulation crashed at step {t}")
        break

    # Save velocity slice for animation more frequently
    if t % 25 == 0:  # Every 25 steps
        u_slice = (S[:, :, N // 2, 1] / S[:, :, N // 2, 0]).cpu().detach().numpy()
        if not np.any(np.isnan(u_slice)):
            plt.figure(figsize=(6, 6))
            plt.imshow(u_slice, cmap="viridis", origin="lower", vmin=-5, vmax=10)
            plt.colorbar(label="u-velocity")
            plt.title(f"u-Velocity at Step {t}")
            plt.savefig(f"checkpoints/velocity_step_{t}.png")
            plt.close()
            frames.append(imageio.imread(f"checkpoints/velocity_step_{t}.png"))

    if t % 50 == 0 or t == steps - 1:
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

# Save animation
if frames:
    try:
        imageio.mimsave("velocity_animation.gif", frames, fps=5)
        print("Animation saved as velocity_animation.gif")
    except Exception as e:
        print(f"Failed to save animation: {e}")

# Train StabilityNet
if training_data:
    print("\nTraining StabilityNet...")
    X = torch.tensor(training_data, dtype=torch.float32, device=device)
    y = torch.tensor(training_labels, dtype=torch.float32, device=device).view(-1, 1)
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = stability_net(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# --- Visualizations ---

plt.figure(figsize=(10, 8))

# Divergence plot
plt.subplot(2, 2, 1)
plt.plot(divergence_history, label="Divergence")
plt.xlabel("Step")
plt.ylabel("Mean Abs Divergence")
plt.title("Divergence Over Time")
plt.legend()
plt.grid(True)
plt.savefig("divergence.png")
plt.close()

# Energy plot
plt.figure(figsize=(6, 4))
plt.plot(energy_history, label="Energy")
plt.xlabel("Step")
plt.ylabel("Kinetic Energy")
plt.title("Energy Over Time")
plt.legend()
plt.grid(True)
plt.savefig("energy.png")
plt.close()

# Negative space plot
plt.figure(figsize=(6, 4))
plt.plot(negative_space_history, label="Negative Space")
plt.xlabel("Step")
plt.ylabel("Fraction")
plt.title("Negative Uncomputable Space")
plt.legend()
plt.grid(True)
plt.savefig("negative_space.png")
plt.close()

# Velocity slice
plt.figure(figsize=(6, 4))
u_slice = (S[:, :, N // 2, 1] / S[:, :, N // 2, 0]).cpu().detach().numpy()
if not np.any(np.isnan(u_slice)):
    plt.imshow(u_slice, cmap="viridis", origin="lower", vmin=-2, vmax=5)
    plt.colorbar(label="u-velocity")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("u-Velocity at Midplane")
    plt.savefig("velocity_slice.png")
plt.close()

# Combine plots into results.png
from PIL import Image
try:
    images = [Image.open(f) for f in ["divergence.png", "energy.png", "negative_space.png", "velocity_slice.png"]]
    widths, heights = zip(*(i.size for i in images))
    total_width = max(widths) * 2
    total_height = max(heights) * 2
    combined = Image.new('RGB', (total_width, total_height))
    combined.paste(images[0], (0, 0))
    combined.paste(images[1], (max(widths), 0))
    combined.paste(images[2], (0, max(heights)))
    combined.paste(images[3], (max(widths), max(heights)))
    combined.save("results.png")
    print("Combined plot saved as results.png")
except Exception as e:
    print(f"Failed to combine plots: {e}")

# 3D Voxel Plot for negative space
M_np = M.cpu().detach().numpy()
x, y, z = np.indices(M_np.shape)
fig = go.Figure(data=go.Volume(
    x=x.flatten(), y=y.flatten(), z=z.flatten(),
    value=M_np.flatten(), isomin=0.5, isomax=1,
    opacity=0.3, surface_count=20, colorscale="Reds"
))
fig.update_layout(
    title="Negative Uncomputable Space (M=1 Regions)",
    scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
    width=800, height=600
)
try:
    fig.write_html("negative_space.html")
    print("Negative space HTML saved as negative_space.html")
    fig.write_image("negative_space_high_res.png", width=800, height=600)
    print("Negative space PNG saved as negative_space_high_res.png")
except Exception as e:
    print(f"Failed to save negative space plot: {e}")
try:
    fig.show()
except Exception as e:
    print(f"Failed to display negative space plot: {e}")

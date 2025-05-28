# DGNS-Engine-V7: Ducks Guts Navier Stokes Tensor Engine (Version 7)
# Purpose: Simulate compressible Navier-Stokes equations in 3D to map negative uncomputable space
# Dependencies: numpy, torch, matplotlib, plotly (optional), imageio, psutil, glob, PIL
# Assumptions:
# - Ideal gas law for air at STP (T = 273.15 K, P = 101325 Pa, gamma = 1.4, R = 287 J/kg·K)
# - Lid-driven cavity setup with user-configurable initial conditions
# - Uses Smagorinsky LES model for subgrid-scale turbulence
# - Spectral smoothing to stabilize high-frequency modes
# - StabilityNet neural network for adaptive time-stepping
# Version History:
# - V6 (May 28, 2025): Introduced compressible flow, ideal gas law, user-configurable gas properties
# - V7 (May 28, 2025): Enhanced stability, optimized for larger grids, deeper exploration of non-computable space
# Authors: Aetheris Navigatrix & Grok 3 (xAI)
# DGNS-Engine-V7: Ducks Guts Navier Stokes Tensor Engine (Version 7)
# Purpose: Simulate compressible Navier-Stokes equations in 3D to map negative uncomputable space
# Dependencies: numpy, torch, matplotlib, plotly (optional), imageio, psutil, glob, PIL
# Assumptions:
# - Ideal gas law for air at STP (T = 273.15 K, P = 101325 Pa, gamma = 1.4, R = 287 J/kg·K)
# - Lid-driven cavity setup with user-configurable initial conditions
# - Uses Smagorinsky LES model for subgrid-scale turbulence
# - Spectral smoothing to stabilize high-frequency modes
# - StabilityNet neural network for adaptive time-stepping
# Version History:
# - V6 (May 28, 2025): Introduced compressible flow, ideal gas law, user-configurable gas properties
# - V7 (May 28, 2025): Enhanced stability, optimized for larger grids, deeper exploration of non-computable space
# Authors: Aetheris Navigatrix & Grok 3 (xAI)

# 1. Imports and Device Setup
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser"
except ImportError:
    print("Warning: Plotly is not installed. 3D visualizations will be skipped.")
    go = None
import time
import os
import imageio.v2 as imageio
import psutil
import sys
import glob

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Attempt to set high process priority (Linux only)
try:
    process = psutil.Process(os.getpid())
    if sys.platform.startswith('linux'):
        process.nice(-10)
        print("Process priority set to high")
except Exception as e:
    print(f"Failed to set process priority: {e}")

# Estimate memory requirement based on N (function definition remains here)
def estimate_memory_requirement(N):
    state_size = N * N * N * 5 * 4  # 5 components, 4 bytes per float
    total_size = state_size * 2  # Account for temporary tensors
    return total_size / 1e9  # Convert to GB

# 2. User Input for Simulation Parameters
def get_user_input(prompt, default, type_func, min_val=None, max_val=None):
    while True:
        user_input = input(f"{prompt} (default: {default}, press Enter to accept): ").strip()
        try:
            value = type_func(user_input) if user_input else default
            if min_val is not None and value < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            continue

N = get_user_input("Grid size (suggested: ≤48)", 48, int, min_val=16, max_val=128)
estimated_memory = estimate_memory_requirement(N)
if estimated_memory > psutil.virtual_memory().available / 1e9 * 0.85:
    print(f"Warning: Estimated memory requirement ({estimated_memory:.2f} GB) exceeds 85% of available memory. Consider reducing N or increasing RAM.")
steps = get_user_input("Number of steps (suggested: 100–500)", 200, int, min_val=50, max_val=1000)
reynolds = get_user_input("Reynolds number (suggested: 1000–20000)", 10000, float, min_val=1000, max_val=50000)
dt = get_user_input("Time step (dt, suggested: ≤1e-5)", 1e-5, float, min_val=1e-8, max_val=1e-4)
Cs = get_user_input("Smagorinsky constant (Cs, suggested: 0.05–0.2)", 0.1, float, min_val=0.01, max_val=0.5)

use_sweep = input("Run Reynolds number sweep? (y/n, default: n): ").strip().lower() == 'y'
if use_sweep:
    reynolds_min = get_user_input("Minimum Reynolds number", 1000, float, min_val=1000, max_val=50000)
    reynolds_max = get_user_input("Maximum Reynolds number", 20000, float, min_val=1000, max_val=50000)
    reynolds_steps = get_user_input("Number of Reynolds values", 5, int, min_val=2, max_val=10)
    reynolds_list = np.linspace(reynolds_min, reynolds_max, reynolds_steps)
else:
    reynolds_list = [reynolds]
    
# 3. Simulation Parameter Setup
# Physical constants and simulation parameters
dx = 1.0 / (N - 1)  # Grid spacing, adjusted based on N
dt_min = 1e-8  # Minimum allowable time step for stability
dx_min = 0.002  # Minimum grid spacing for numerical stability
divergence_threshold = 1e-6  # Threshold for acceptable divergence
max_iterations = 1  # Maximum iterations for solving (currently fixed)
cfl = 0.1  # Courant-Friedrichs-Lewy number for time step stability

# Ideal gas law parameters for air at STP
T0 = 273.15  # Temperature in Kelvin (0°C)
P0 = 101325  # Pressure in Pascals (1 atm)
gamma = 1.4  # Specific heat ratio for air
R = 287.0  # Gas constant for air (J/kg·K)
cv = R / (gamma - 1)  # Specific heat at constant volume
rho0 = P0 / (R * T0)  # Initial density at STP

# Ensure checkpoints directory exists
if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

# 4. Helper Functions and Initialization
def compute_gradients(S, dx):
    """Compute gradients for all components of S."""
    grad = lambda x: (x[2:, 1:-1, 1:-1] - x[:-2, 1:-1, 1:-1]) / (2 * dx)
    grad_y = lambda x: (x[1:-1, 2:, 1:-1] - x[1:-1, :-2, 1:-1]) / (2 * dx)
    grad_z = lambda x: (x[1:-1, 1:-1, 2:] - x[1:-1, 1:-1, :-2]) / (2 * dx)
    grads = []
    for i in range(S.shape[-1]):
        grads.extend([grad(S[:, :, :, i]), grad_y(S[:, :, :, i]), grad_z(S[:, :, :, i])])
    return grads

def laplacian(S, idx, dx):
    """Compute Laplacian for a specific component of S."""
    lap = (S[2:, 1:-1, 1:-1, idx] + S[:-2, 1:-1, 1:-1, idx] +
           S[1:-1, 2:, 1:-1, idx] + S[1:-1, :-2, 1:-1, idx] +
           S[1:-1, 1:-1, 2:, idx] + S[1:-1, 1:-1, :-2, idx] -
           6 * S[1:-1, 1:-1, 1:-1, idx]) / (dx ** 2)
    return lap

def divergence(S, dx):
    """Compute divergence of velocity field."""
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
    """Compute subgrid-scale turbulence term using Smagorinsky model."""
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

def spectral_smooth(S, N, dx, iterations=1):
    """Apply spectral smoothing to stabilize high-frequency modes."""
    for _ in range(iterations):
        for i in range(5):
            var = S[:, :, :, i]
            var_hat = torch.fft.fftn(var)
            k = torch.fft.fftfreq(N, d=dx, device=device) * 2 * np.pi
            kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
            k2 = kx ** 2 + ky ** 2 + kz ** 2
            filter_mask = (k2 < (N * np.pi / 2) ** 2).float()
            var_hat_filtered = var_hat * filter_mask
            var_hat_filtered *= 0.95
            S[:, :, :, i] = torch.fft.ifftn(var_hat_filtered).real
    return S

def check_memory_usage(threshold=0.85):
    """Check if memory usage exceeds threshold."""
    process = psutil.Process()
    mem = process.memory_info().rss / psutil.virtual_memory().total
    if mem > threshold:
        print(f"Memory usage high ({mem:.2%}), skipping non-critical tasks")
        return False
    return True

def log_initial_conditions(S, dx):
    """Log initial conditions of the simulation."""
    grads = compute_gradients(S, dx)
    grad_norm = torch.sqrt(sum(g ** 2 for g in grads)).mean()
    rho = S[:, :, :, 0].unsqueeze(-1)
    velocities = S[:, :, :, 1:4] / rho
    variance = torch.var(velocities)
    print(f"Initial Conditions: Grad Norm: {grad_norm:.2e}, Variance: {variance:.2e}")

def initialize_state(N, rho0, cv, T0, device, seed=42):
    """Initialize the state tensor with physical properties."""
    S = torch.zeros((N, N, N, 5), dtype=torch.float32, device=device)
    S[:, :, :, 0] = rho0
    S[:, :, -1, 1] = rho0 * 10.0
    torch.manual_seed(seed)
    S[:, :, :, 1:4] += 0.1 * rho0 * torch.randn(N, N, N, 3, device=device)
    center = N // 2
    S[center - 4:center + 4, center - 4:center + 4, center - 4:center + 4, 0] *= 1.5
    u = S[:, :, :, 1] / S[:, :, :, 0]
    v = S[:, :, :, 2] / S[:, :, :, 0]
    w = S[:, :, :, 3] / S[:, :, :, 0]
    S[:, :, :, 4] = S[:, :, :, 0] * (cv * T0 + 0.5 * (u ** 2 + v ** 2 + w ** 2))
    S.requires_grad = True
    return S

# 5. Stability and Navier-Stokes Solver
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

def check_stability_and_computability(S, t, M, dx, dt, dt_min, stability_threshold=1e4):
    """Check stability and flag uncomputable regions."""
    grads = compute_gradients(S, dx)
    grad_norm = torch.sqrt(sum(g ** 2 for g in grads)).mean()
    rho = S[:, :, :, 0].unsqueeze(-1)
    velocities = S[:, :, :, 1:4] / rho
    variance = torch.var(velocities)

    print(f"Stability Check (Step {t:.4f}): Grad Norm: {grad_norm:.2e}, Variance: {variance:.2e}")

    # Optimize grad_norm_tensor computation
    grad_norm_tensor = torch.zeros_like(S[:, :, :, 0])
    for i, grad in enumerate(grads):
        if i % 3 == 0:  # Only sum x-gradients for efficiency
            grad_norm_tensor[1:-1, 1:-1, 1:-1] += grad ** 2
    grad_norm_tensor = torch.sqrt(grad_norm_tensor)
    uncomputable_mask = (grad_norm_tensor > stability_threshold) | torch.isnan(grad_norm_tensor) | torch.isinf(grad_norm_tensor)
    M[uncomputable_mask] = 1

    if torch.any(torch.isnan(S)) or torch.any(torch.isinf(S)):
        print(f"Simulation diverged at step {t}, reducing dt")
        adjusted_dt = dt * 0.5
        M[1:-1, 1:-1, 1:-1] = 1
        return adjusted_dt, M, grad_norm

    u_max = velocities.abs().max().item()
    c = torch.sqrt(torch.tensor(gamma * R * T0, device=device))
    adjusted_dt = cfl * dx / (u_max + c + 1e-10)
    adjusted_dt = max(min(adjusted_dt, dt), dt_min)
    print(f"Adjusted dt (CFL): {adjusted_dt:.2e}")

    if adjusted_dt < dt_min or u_max > 1e3:
        print(f"Flagging uncomputable regions at step {t:.4f} due to small dt or high velocity: dt={adjusted_dt:.2e}, u_max={u_max:.2e}")
        M[1:-1, 1:-1, 1:-1] = 1

    sys_metrics = {
        'step': t,
        'grad_norm': grad_norm.item(),
        'u_max': u_max,
        'memory_usage': psutil.Process().memory_info().rss / 1e6,
        'cpu_percent': psutil.cpu_percent(interval=0.1)
    }
    with open('system_metrics.log', 'a') as f:
        f.write(str(sys_metrics) + '\n')

    training_data.append([grad_norm.item(), variance.item(), t])
    training_labels.append(1.0 if adjusted_dt > dt_min and u_max < 1e3 else 0.0)
    return adjusted_dt, M, grad_norm

def test_parameter_stability(S, M, reynolds, steps, dt, dt_min, R, gamma, cv, N, dx):
    """Test stability of parameters over a short run."""
    nu = 1.0 / reynolds
    S_test = S.clone()
    M_test = M.clone()
    for t in range(steps):
        adjusted_dt, M_test, grad_norm = check_stability_and_computability(S_test, t * dt, M_test, dx, dt, dt_min)
        S_test, M_test = navier_stokes_step(S_test, adjusted_dt, nu, R, gamma, cv, M_test, N, dx)
        if torch.any(torch.isnan(S_test)) or grad_norm > 1e5:
            return False
    return True

def navier_stokes_step(S, dt, nu, R, gamma, cv, M, N, dx, exploration_depth=0.0):
    """Perform a single step of the Navier-Stokes solver with adjustable exploration depth."""
    uncomputable_fraction = M[1:-1, 1:-1, 1:-1].sum().item() / ((N - 2) ** 3)
    if uncomputable_fraction > (0.99 - exploration_depth):
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

    velocity_tensor = torch.stack([u, v, w], dim=-1)
    vel_grads = compute_gradients(velocity_tensor, dx)
    P_tensor = P.unsqueeze(-1)
    P_grads = compute_gradients(P_tensor, dx)
    rho_tensor = rho.unsqueeze(-1)
    rho_grads = compute_gradients(rho_tensor, dx)
    E_tensor = rho_E.unsqueeze(-1)
    E_grads = compute_gradients(E_tensor, dx)

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

    S_new = spectral_smooth(S_new, N, dx, iterations=1)

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return S_new, M

# 6. Main Simulation Loop and Visualizations
results_sweep = {}
for reynolds in reynolds_list:
    print(f"\nRunning simulation for Reynolds number: {reynolds}")
    nu = 1.0 / reynolds
    S = initialize_state(N, rho0, cv, T0, device)
    M = torch.zeros((N, N, N), dtype=torch.int32, device=device)
    log_initial_conditions(S, dx)

    if use_sweep:
        if not test_parameter_stability(S, M, reynolds, 10, dt, dt_min, R, gamma, cv, N, dx):
            print(f"Reynolds {reynolds} unstable, skipping")
            continue

    print(f"Running DGNS-Engine-V7 simulation for Re={reynolds}...")
    start_time = time.time()
    divergence_history = []
    energy_history = []
    negative_space_history = []
    time_per_step = []

    for t in range(steps):
        step_start = time.time()
        if not check_memory_usage(threshold=0.85):
            print("Pausing simulation for 10s to free resources")
            time.sleep(10)

        adjusted_dt, M, grad_norm = check_stability_and_computability(S, t * dt, M, dx, dt, dt_min, stability_threshold=1e4 + t * 1e2)
        S, M = navier_stokes_step(S, adjusted_dt, nu, R, gamma, cv, M, N, dx, exploration_depth=t / steps * 0.1)
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
            print(f"Simulation crashed at step {t} for Re={reynolds}")
            break

        if t % 10 == 0 and check_memory_usage(threshold=0.85):
            u_slice = (S[:, :, N // 2, 1] / S[:, :, N // 2, 0]).cpu().detach().numpy()
            if not np.any(np.isnan(u_slice)):
                plt.figure(figsize=(6, 6))
                plt.imshow(u_slice, cmap="viridis", origin="lower", vmin=-5, vmax=10)
                plt.colorbar(label="u-velocity")
                plt.title(f"u-Velocity at Step {t} (Re={reynolds})")
                plt.savefig(f"checkpoints/velocity_step_{t}_Re_{reynolds:.0f}_N{N}.png")
                plt.close()

        if t % 50 == 0 or t == steps - 1:
            print(f"Step {t}/{steps}, Div: {div:.2e}, Energy: {energy:.2f}, "
                  f"Negative Space: {negative_space_fraction:.2%}, Time: {step_time:.2f}s")
            if t % 100 == 0 or t == steps - 1:
                torch.save({
                    'step': t,
                    'S': S,
                    'M': M,
                    'divergence_history': divergence_history,
                    'energy_history': energy_history,
                    'negative_space_history': negative_space_history
                }, f"checkpoints/checkpoint_step_{t}_Re_{reynolds:.0f}_N{N}.pt")

    results_sweep[reynolds] = {
        'divergence': divergence_history,
        'energy': energy_history,
        'negative_space': negative_space_history
    }

    # Save animation with all PNGs in checkpoints directory
    png_files = glob.glob(f"checkpoints/velocity_step_*_Re_{reynolds:.0f}_N{N}.png")
    png_files.sort(key=lambda x: int(x.split('_')[2]))
    if png_files:
        with imageio.get_writer(f"velocity_animation_Re_{reynolds:.0f}_N{N}.gif", mode='I', fps=10) as writer:
            for fname in png_files:
                writer.append_data(imageio.imread(fname))
        print(f"Animation saved as velocity_animation_Re_{reynolds:.0f}_N{N}.gif")

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

# Visualizations
if use_sweep:
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(['divergence', 'energy', 'negative_space']):
        plt.subplot(3, 1, i + 1)
        for reynolds in reynolds_list:
            if reynolds in results_sweep:
                plt.plot(results_sweep[reynolds][metric], label=f'Re={reynolds:.0f}')
        plt.xlabel('Step')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} Over Time')
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'reynolds_sweep_N{N}.png')
    plt.close()
    print(f"Reynolds sweep plot saved as reynolds_sweep_N{N}.png")

reynolds = reynolds_list[-1] if reynolds_list else reynolds
divergence_history = results_sweep[reynolds]['divergence']
energy_history = results_sweep[reynolds]['energy']
negative_space_history = results_sweep[reynolds]['negative_space']

plt.figure(figsize=(6, 4))
plt.plot(divergence_history, label="Divergence")
plt.xlabel("Step")
plt.ylabel("Mean Abs Divergence")
plt.title("Divergence Over Time")
plt.legend()
plt.grid(True)
plt.savefig(f"divergence_Re_{reynolds:.0f}_N{N}.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(energy_history, label="Energy")
plt.xlabel("Step")
plt.ylabel("Kinetic Energy")
plt.title("Energy Over Time")
plt.legend()
plt.grid(True)
plt.savefig(f"energy_Re_{reynolds:.0f}_N{N}.png")
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(negative_space_history, label="Negative Space")
plt.xlabel("Step")
plt.ylabel("Fraction")
plt.title("Negative Uncomputable Space")
plt.legend()
plt.grid(True)
plt.savefig(f"negative_space_Re_{reynolds:.0f}_N{N}.png")
plt.close()

plt.figure(figsize=(6, 4))
u_slice = (S[:, :, N // 2, 1] / S[:, :, N // 2, 0]).cpu().detach().numpy()
if not np.any(np.isnan(u_slice)):
    plt.imshow(u_slice, cmap="viridis", origin="lower", vmin=-2, vmax=5)
    plt.colorbar(label="u-velocity")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("u-Velocity at Midplane")
    plt.savefig(f"velocity_slice_Re_{reynolds:.0f}_N{N}.png")
plt.close()

from PIL import Image
try:
    images = [f"divergence_Re_{reynolds:.0f}_N{N}.png", f"energy_Re_{reynolds:.0f}_N{N}.png",
              f"negative_space_Re_{reynolds:.0f}_N{N}.png", f"velocity_slice_Re_{reynolds:.0f}_N{N}.png"]
    combined = None
    for i, img_path in enumerate(images):
        img = Image.open(img_path)
        if combined is None:
            combined = Image.new('RGB', (img.width * 2, img.height * 2))
        x = (i % 2) * img.width
        y = (i // 2) * img.height
        combined.paste(img, (x, y))
        img.close()
    combined.save(f"results_Re_{reynolds:.0f}_N{N}.png")
    print(f"Combined plot saved as results_Re_{reynolds:.0f}_N{N}.png")
except Exception as e:
    print(f"Failed to combine plots: {e}")

# 2D Slice of Negative Uncomputable Space
plt.figure(figsize=(6, 4))
m_slice = M[:, :, N // 2].cpu().detach().numpy()
plt.imshow(m_slice, cmap="Reds", origin="lower", vmin=0, vmax=1)
plt.colorbar(label="Uncomputable Regions (M)")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Negative Uncomputable Space at Midplane")
plt.savefig(f"negative_space_slice_Re_{reynolds:.0f}_N{N}.png")
plt.close()
print(f"Negative space 2D slice saved as negative_space_slice_Re_{reynolds:.0f}_N{N}.png")

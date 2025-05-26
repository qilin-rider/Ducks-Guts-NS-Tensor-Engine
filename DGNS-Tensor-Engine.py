import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time

# Set device (xAI HyperCluster: 64 NVIDIA H100 GPUs, 141 GB each)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simulation parameters
N = 64  # Coarse grid size
dx = 1.0 / (N - 1)  # Coarse spatial step size
dt = 0.001  # Initial time step
rho = 1.0  # Density
steps = 1000  # Number of time steps per iteration
dt_min = 1e-7  # Minimum allowable time step
dx_min = 0.004  # Minimum grid size (equivalent to 256^3)
divergence_threshold = 1e-5  # Threshold to switch to DST
Cs = 0.1  # Smagorinsky constant for LES
max_iterations = 2  # Maximum refinement iterations

# Initialize the state tensor S: [u, v, w, P] on a 3D grid
S = torch.zeros((N, N, N, 4), dtype=torch.float32, device=device)
S[:, :, -1, 0] = 1.0  # Lid-driven cavity: u = 1 at z = 1
S.requires_grad = True

# Metadata tensor to track computability
M = torch.zeros((N, N, N), dtype=torch.int32, device=device)

# Stability network
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

# Training data for StabilityNet
training_data = []
training_labels = []

# Finite difference helpers
def compute_gradients(S, dx):
    grad_u_x = (S[2:, 1:-1, 1:-1, 0] - S[:-2, 1:-1, 1:-1, 0]) / (2 * dx)
    grad_u_y = (S[1:-1, 2:, 1:-1, 0] - S[1:-1, :-2, 1:-1, 0]) / (2 * dx)
    grad_u_z = (S[1:-1, 1:-1, 2:, 0] - S[1:-1, 1:-1, :-2, 0]) / (2 * dx)
    grad_v_x = (S[2:, 1:-1, 1:-1, 1] - S[:-2, 1:-1, 1:-1, 1]) / (2 * dx)
    grad_v_y = (S[1:-1, 2:, 1:-1, 1] - S[1:-1, :-2, 1:-1, 1]) / (2 * dx)
    grad_v_z = (S[1:-1, 1:-1, 2:, 1] - S[1:-1, 1:-1, :-2, 1]) / (2 * dx)
    grad_w_x = (S[2:, 1:-1, 1:-1, 2] - S[:-2, 1:-1, 1:-1, 2]) / (2 * dx)
    grad_w_y = (S[1:-1, 2:, 1:-1, 2] - S[1:-1, :-2, 1:-1, 2]) / (2 * dx)
    grad_w_z = (S[1:-1, 1:-1, 2:, 2] - S[1:-1, 1:-1, :-2, 2]) / (2 * dx)
    grad_P_x = (S[2:, 1:-1, 1:-1, 3] - S[:-2, 1:-1, 1:-1, 3]) / (2 * dx)
    grad_P_y = (S[1:-1, 2:, 1:-1, 3] - S[1:-1, :-2, 1:-1, 3]) / (2 * dx)
    grad_P_z = (S[1:-1, 1:-1, 2:, 3] - S[1:-1, 1:-1, :-2, 3]) / (2 * dx)
    return grad_u_x, grad_u_y, grad_u_z, grad_v_x, grad_v_y, grad_v_z, grad_w_x, grad_w_y, grad_w_z, grad_P_x, grad_P_y, grad_P_z

def laplacian(S, idx, dx):
    lap = (S[2:, 1:-1, 1:-1, idx] + S[:-2, 1:-1, 1:-1, idx] +
           S[1:-1, 2:, 1:-1, idx] + S[1:-1, :-2, 1:-1, idx] +
           S[1:-1, 1:-1, 2:, idx] + S[1:-1, 1:-1, :-2, idx] -
           6 * S[1:-1, 1:-1, 1:-1, idx]) / (dx**2)
    return lap

def divergence(S, dx):
    grad_u_x, grad_u_y, grad_u_z, grad_v_x, grad_v_y, grad_v_z, grad_w_x, grad_w_y, grad_w_z, _, _, _ = compute_gradients(S, dx)
    div = grad_u_x + grad_v_y + grad_w_z
    return div

# FFT pressure solver
def solve_pressure_fft(S, rho, dt, N, dx):
    div = divergence(S, dx)
    div_hat = torch.fft.fftn(div)
    k = torch.fft.fftfreq(N, d=dx, device=device) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0  # Avoid division by zero
    P_hat = (rho / dt) * div_hat / k2
    P_hat[0, 0, 0] = 0.0
    P = torch.fft.ifftn(P_hat).real
    S_new = S.clone()
    S_new[1:-1, 1:-1, 1:-1, 3] = P
    return S_new

# DST-like pressure solver (finite difference proxy)
def solve_pressure_dst(S, rho, dt, N, dx, iterations=50):
    div = divergence(S, dx)
    P = S[1:-1, 1:-1, 1:-1, 3].clone()
    for _ in range(iterations):
        P_new = (P[2:, 1:-1, 1:-1] + P[:-2, 1:-1, 1:-1] +
                 P[1:-1, 2:, 1:-1] + P[1:-1, :-2, 1:-1] +
                 P[1:-1, 1:-1, 2:] + P[1:-1, 1:-1, :-2] -
                 (dx**2) * rho * div / dt) / 6.0
        P = P_new
    S_new = S.clone()
    S_new[1:-1, 1:-1, 1:-1, 3] = P
    return S_new

# LES subgrid-scale model (Smagorinsky)
def compute_sgs_term(S, dx):
    grad_u_x, grad_u_y, grad_u_z, grad_v_x, grad_v_y, grad_v_z, grad_w_x, grad_w_y, grad_w_z, _, _, _ = compute_gradients(S, dx)
    Sxx = grad_u_x
    Syy = grad_v_y
    Szz = grad_w_z
    Sxy = 0.5 * (grad_u_y + grad_v_x)
    Syz = 0.5 * (grad_v_z + grad_w_y)
    Sxz = 0.5 * (grad_u_z + grad_w_x)
    S_mag = torch.sqrt(2 * (Sxx**2 + Syy**2 + Szz**2 + 2 * Sxy**2 + 2 * Syz**2 + 2 * Sxz**2))
    nu_t = (Cs * dx)**2 * S_mag  # Eddy viscosity
    return nu_t

# Spectral smoothing
def spectral_smooth(S, N, dx):
    for i in range(3):
        vel = S[:, :, :, i]
        vel_hat = torch.fft.fftn(vel)
        k = torch.fft.fftfreq(N, d=dx, device=device) * 2 * np.pi
        kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
        k2 = kx**2 + ky**2 + kz**2
        filter_mask = (k2 < (N * np.pi / 2)**2).float()
        vel_hat_filtered = vel_hat * filter_mask
        vel_filtered = torch.fft.ifftn(vel_hat_filtered).real
        S[:, :, :, i] = vel_filtered
    return S

# Check stability and computability
def check_stability_and_computability(S, t, M, dx):
    grad_u_x, grad_u_y, grad_u_z, grad_v_x, grad_v_y, grad_v_z, grad_w_x, grad_w_y, grad_w_z, _, _, _ = compute_gradients(S, dx)
    grad_norm = torch.sqrt(grad_u_x**2 + grad_u_y**2 + grad_u_z**2 +
                           grad_v_x**2 + grad_v_y**2 + grad_v_z**2 +
                           grad_w_x**2 + grad_w_y**2 + grad_w_z**2).mean()
    variance = torch.var(S[:, :, :, 0:3])
    
    # Additional check for high Re: cap gradient norm to prevent blow-up
    if grad_norm > 1e3:  # Arbitrary threshold for Re=10000
        M[1:-1, 1:-1, 1:-1] = 1
        adjusted_dt = dt_min
    else:
        stability_input = torch.tensor([grad_norm.item(), variance.item(), t], dtype=torch.float32, device=device)
        adjustment = stability_net(stability_input)
        adjusted_dt = adjustment * dt
        if adjusted_dt < dt_min:
            M[1:-1, 1:-1, 1:-1] = 1
            adjusted_dt = dt_min

    training_data.append([grad_norm.item(), variance.item(), t])
    training_labels.append(1.0 if adjusted_dt > dt_min else 0.0)

    return adjusted_dt, M, grad_norm

# Navier-Stokes step with LES
def navier_stokes_step(S, dt, nu, rho, M, N, dx, use_dst=False):
    if M[1:-1, 1:-1, 1:-1].sum() == (N-2)**3:
        return S, M, False

    u, v, w, P = S[:, :, :, 0], S[:, :, :, 1], S[:, :, :, 2], S[:, :, :, 3]
    grad_u_x, grad_u_y, grad_u_z, grad_v_x, grad_v_y, grad_v_z, grad_w_x, grad_w_y, grad_w_z, grad_P_x, grad_P_y, grad_P_z = compute_gradients(S, dx)

    # Nonlinear convective terms
    conv_u = u[1:-1, 1:-1, 1:-1] * grad_u_x + v[1:-1, 1:-1, 1:-1] * grad_u_y + w[1:-1, 1:-1, 1:-1] * grad_u_z
    conv_v = u[1:-1, 1:-1, 1:-1] * grad_v_x + v[1:-1, 1:-1, 1:-1] * grad_v_y + w[1:-1, 1:-1, 1:-1] * grad_v_z
    conv_w = u[1:-1, 1:-1, 1:-1] * grad_w_x + v[1:-1, 1:-1, 1:-1] * grad_w_y + w[1:-1, 1:-1, 1:-1] * grad_w_z

    # Viscous terms with LES
    nu_t = compute_sgs_term(S, dx)
    effective_nu = nu + nu_t
    lap_u = laplacian(S, 0, dx)
    lap_v = laplacian(S, 1, dx)
    lap_w = laplacian(S, 2, dx)

    # Update velocities
    u_new = u[1:-1, 1:-1, 1:-1] + dt * (-conv_u + effective_nu[1:-1, 1:-1, 1:-1] * lap_u)
    v_new = v[1:-1, 1:-1, 1:-1] + dt * (-conv_v + effective_nu[1:-1, 1:-1, 1:-1] * lap_v)
    w_new = w[1:-1, 1:-1, 1:-1] + dt * (-conv_w + effective_nu[1:-1, 1:-1, 1:-1] * lap_w)

    S_new = S.clone()
    S_new[1:-1, 1:-1, 1:-1, 0] = torch.where(M[1:-1, 1:-1, 1:-1] == 0, u_new, S[1:-1, 1:-1, 1:-1, 0])
    S_new[1:-1, 1:-1, 1:-1, 1] = torch.where(M[1:-1, 1:-1, 1:-1] == 0, v_new, S[1:-1, 1:-1, 1:-1, 1])
    S_new[1:-1, 1:-1, 1:-1, 2] = torch.where(M[1:-1, 1:-1, 1:-1] == 0, w_new, S[1:-1, 1:-1, 1:-1, 2])

    # Solve pressure
    div = divergence(S_new, dx)
    use_dst = (div.abs().max() > divergence_threshold)
    if use_dst:
        S_new = solve_pressure_dst(S_new, rho, dt, N, dx)
    else:
        S_new = solve_pressure_fft(S_new, rho, dt, N, dx)

    # Correct velocities
    _, _, _, _, _, _, _, _, _, grad_P_x, grad_P_y, grad_P_z = compute_gradients(S_new, dx)
    S_new[1:-1, 1:-1, 1:-1, 0] -= dt * grad_P_x / rho
    S_new[1:-1, 1:-1, 1:-1, 1] -= dt * grad_P_y / rho
    S_new[1:-1, 1:-1, 1:-1, 2] -= dt * grad_P_z / rho

    S_new = spectral_smooth(S_new, N, dx)
    return S_new, M, use_dst

# Iterative refinement on sub-grids
def refine_subgrid(S, M, N_coarse, dx_coarse, N_fine, dx_fine, region, nu, rho):
    # Extract region (simplified: assume region is a cube near the lid)
    i_start, i_end, j_start, j_end, k_start, k_end = region
    S_region = S[i_start:i_end, j_start:j_end, k_start:k_end, :].clone()
    M_region = M[i_start:i_end, j_start:j_end, k_start:k_end].clone()

    # Interpolate coarse solution to fine grid
    S_fine = torch.nn.functional.interpolate(S_region.permute(3, 0, 1, 2), 
                                             size=(N_fine, N_fine, N_fine), 
                                             mode='trilinear', align_corners=True)
    S_fine = S_fine.permute(1, 2, 3, 0)
    M_fine = torch.zeros((N_fine, N_fine, N_fine), dtype=torch.int32, device=device)

    # Solve on fine grid
    dt_fine = dt * (dx_fine / dx_coarse)**2  # Scale time step with grid size
    for t in range(steps):
        adjusted_dt, M_fine, _ = check_stability_and_computability(S_fine, t * dt_fine, M_fine, dx_fine)
        S_fine, M_fine, _ = navier_stokes_step(S_fine, adjusted_dt, nu, rho, M_fine, N_fine, dx_fine)
        if torch.any(torch.isnan(S_fine)):
            break

    # Interpolate back to coarse grid
    S_coarse = torch.nn.functional.interpolate(S_fine.permute(3, 0, 1, 2), 
                                               size=(i_end-i_start, j_end-j_start, k_end-k_start), 
                                               mode='trilinear', align_corners=True)
    S_coarse = S_coarse.permute(1, 2, 3, 0)
    S[i_start:i_end, j_start:j_end, k_start:k_end, :] = S_coarse
    M[i_start:i_end, j_start:j_end, k_start:k_end] = (M_fine[::N_fine//(i_end-i_start), ::N_fine//(j_end-j_start), ::N_fine//(k_end-k_start)] > 0).int()

    return S, M

# Main simulation with iterative refinement
reynolds_number = 10000  # Focus on high Re
nu = 1.0 / reynolds_number
results = {}

# First pass: Coarse grid
print(f"\nFirst Pass: Coarse grid (Re={reynolds_number})")
divergence_history = []
energy_history = []
negative_space_history = []
time_per_step = []

start_time = time.time()
for t in range(steps):
    step_start = time.time()
    adjusted_dt, M, grad_norm = check_stability_and_computability(S, t * dt, M, dx)
    S, M, use_dst = navier_stokes_step(S, adjusted_dt, nu, rho, M, N, dx)
    step_time = time.time() - step_start
    time_per_step.append(step_time)

    div = divergence(S, dx).abs().mean().item()
    energy = 0.5 * (S[:, :, :, 0:3]**2).sum().item()  # Kinetic energy
    negative_space_fraction = M.sum().item() / (N**3)
    divergence_history.append(div)
    energy_history.append(energy)
    negative_space_history.append(negative_space_fraction)

    if torch.any(torch.isnan(S)):
        print(f"Simulation crashed at step {t}")
        break

    if t % 200 == 0:
        print(f"Step {t}, Divergence: {div:.2e}, Energy: {energy:.2f}, Negative Space: {negative_space_fraction:.2%}, Time/Step: {step_time:.3f}s")

total_time = time.time() - start_time
results["coarse"] = {
    "divergence": divergence_history,
    "energy": energy_history,
    "negative_space": negative_space_history,
    "total_time": total_time,
    "avg_time_per_step": np.mean(time_per_step),
    "final_S": S.clone(),
    "final_M": M.clone()
}

# Iterative refinement: Focus on boundary regions
negative_space_prev = negative_space_history[-1]
for iteration in range(max_iterations):
    print(f"\nRefinement Iteration {iteration + 1}")
    
    # Define refinement levels
    if iteration == 0:
        N_fine = 128  # First refinement: 128^3
    else:
        N_fine = 256  # Second refinement: 256^3
    dx_fine = 1.0 / (N_fine - 1)

    # Stop if grid size reaches minimum
    if dx_fine < dx_min:
        print(f"Reached minimum grid size (dx={dx_fine:.4f} < {dx_min:.4f}), stopping refinement.")
        break

    # Identify boundary region (simplified: near the lid, z >= 0.75)
    region = (0, N, 0, N, int(0.75 * N), N)  # i_start, i_end, j_start, j_end, k_start, k_end
    S, M = refine_subgrid(S, M, N, dx, N_fine, dx_fine, region, nu, rho)

    # Run a few steps on the coarse grid to stabilize
    for t in range(100):
        adjusted_dt, M, _ = check_stability_and_computability(S, t * dt, M, dx)
        S, M, _ = navier_stokes_step(S, adjusted_dt, nu, rho, M, N, dx)

    # Update diagnostics
    div = divergence(S, dx).abs().mean().item()
    energy = 0.5 * (S[:, :, :, 0:3]**2).sum().item()
    negative_space_fraction = M.sum().item() / (N**3)
    divergence_history.append(div)
    energy_history.append(energy)
    negative_space_history.append(negative_space_fraction)

    # Check if negative HD space is still shrinking
    if negative_space_fraction >= negative_space_prev * 0.99:  # Less than 1% improvement
        print(f"Negative HD space not shrinking significantly ({negative_space_fraction:.2%} vs {negative_space_prev:.2%}), stopping refinement.")
        break
    negative_space_prev = negative_space_fraction

    print(f"Iteration {iteration + 1} complete, Negative Space: {negative_space_fraction:.2%}")

# Train StabilityNet
if len(training_data) > 0:
    print("\nTraining StabilityNet...")
    X = torch.tensor(training_data, dtype=torch.float32, device=device)
    y = torch.tensor(training_labels, dtype=torch.float32, device=device).view(-1, 1)
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = stability_net(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Plot results
plt.figure(figsize=(15, 15))

# Divergence history
plt.subplot(2, 2, 1)
plt.plot(results["coarse"]["divergence"], label="Coarse Grid")
plt.plot(range(steps, len(divergence_history)), divergence_history[steps:], label="After Refinement", linestyle="--")
plt.xlabel("Step")
plt.ylabel("Mean Absolute Divergence")
plt.title("Divergence Over Time")
plt.legend()
plt.grid(True)

# Energy history
plt.subplot(2, 2, 2)
plt.plot(results["coarse"]["energy"], label="Coarse Grid")
plt.plot(range(steps, len(energy_history)), energy_history[steps:], label="After Refinement", linestyle="--")
plt.xlabel("Step")
plt.ylabel("Kinetic Energy")
plt.title("Energy Over Time")
plt.legend()
plt.grid(True)

# Negative HD space
plt.subplot(2, 2, 3)
plt.plot(results["coarse"]["negative_space"], label="Coarse Grid")
plt.plot(range(steps, len(negative_space_history)), negative_space_history[steps:], label="After Refinement", linestyle="--")
plt.xlabel("Step")
plt.ylabel("Fraction of Negative HD Space")
plt.title("Negative HD Space Over Time")
plt.legend()
plt.grid(True)

# Velocity field (u-component) at mid-plane
plt.subplot(2, 2, 4)
u_slice = S[:, :, N//2, 0].cpu().detach().numpy()  # u-component at z = N/2
plt.imshow(u_slice, cmap="viridis", origin="lower")
plt.colorbar(label="u-velocity")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"u-Velocity at z = {N//2}")

plt.tight_layout()
plt.show()

print(f"\nSimulation completed. Total time: {total_time:.2f}s, Average time per step: {np.mean(time_per_step):.3f}s")

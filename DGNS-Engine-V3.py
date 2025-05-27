import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

# Set device for Colab GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Simulation parameters
N = 64  # Coarse grid size
dx = 1.0 / (N - 1)
dt = 0.001
rho = 1.0
steps = 300  # Reduced for faster testing
dt_min = 1e-7
dx_min = 0.004
divergence_threshold = 1e-5
Cs = 0.1
max_iterations = 1

# Initialize state tensor with turbulent perturbation
S = torch.zeros((N, N, N, 4), dtype=torch.float32, device=device)
S[:, :, -1, 0] = 1.0  # Lid-driven cavity
torch.manual_seed(42)
S[:, :, :, 0:3] += 0.1 * torch.randn(N, N, N, 3, device=device)
S.requires_grad = True

# Metadata tensor
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

training_data = []
training_labels = []

# Finite difference helpers
def compute_gradients(S, dx):
    grad = lambda x: (x[2:, 1:-1, 1:-1] - x[:-2, 1:-1, 1:-1]) / (2 * dx)
    grad_y = lambda x: (x[1:-1, 2:, 1:-1] - x[1:-1, :-2, 1:-1]) / (2 * dx)
    grad_z = lambda x: (x[1:-1, 1:-1, 2:] - x[1:-1, 1:-1, :-2]) / (2 * dx)
    
    grads = []
    for i in range(4):
        grads.extend([grad(S[:, :, :, i]), grad_y(S[:, :, :, i]), grad_z(S[:, :, :, i])])
    return grads

def laplacian(S, idx, dx):
    lap = (S[2:, 1:-1, 1:-1, idx] + S[:-2, 1:-1, 1:-1, idx] +
           S[1:-1, 2:, 1:-1, idx] + S[1:-1, :-2, 1:-1, idx] +
           S[1:-1, 1:-1, 2:, idx] + S[1:-1, 1:-1, :-2, idx] -
           6 * S[1:-1, 1:-1, 1:-1, idx]) / (dx**2)
    return lap

def divergence(S, dx):
    grad_u_x, _, _, _, grad_v_y, _, _, _, grad_w_z, _, _, _ = compute_gradients(S, dx)
    div = grad_u_x + grad_v_y + grad_w_z
    div_full = torch.zeros_like(S[:, :, :, 0])
    div_full[1:-1, 1:-1, 1:-1] = div
    return div_full

# Pressure solvers
def solve_pressure_fft(S, rho, dt, N, dx):
    div = divergence(S, dx)[1:-1, 1:-1, 1:-1]
    div_hat = torch.fft.fftn(div)
    k = torch.fft.fftfreq(N-2, d=dx, device=device) * 2 * np.pi
    kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
    k2 = kx**2 + ky**2 + kz**2
    k2[0, 0, 0] = 1.0
    P_hat = (rho / dt) * div_hat / k2
    P_hat[0, 0, 0] = 0.0
    P = torch.fft.ifftn(P_hat).real
    S_new = S.clone()
    S_new[1:-1, 1:-1, 1:-1, 3] = P
    return S_new

def solve_pressure_dst(S, rho, dt, N, dx, iterations=50):
    div = divergence(S, dx)[1:-1, 1:-1, 1:-1]  # Shape: [62, 62, 62]
    P = torch.zeros_like(div)  # Initialize P with same shape
    for _ in range(iterations):
        P_lap = (P[2:, 1:-1, 1:-1] + P[:-2, 1:-1, 1:-1] +
                 P[1:-1, 2:, 1:-1] + P[1:-1, :-2, 1:-1] +
                 P[1:-1, 1:-1, 2:] + P[1:-1, 1:-1, :-2] -
                 6 * P[1:-1, 1:-1, 1:-1]) / (dx**2)
        P_new = P[1:-1, 1:-1, 1:-1] - (dx**2) * (rho * div[1:-1, 1:-1, 1:-1] / dt - P_lap) / 6.0
        P[1:-1, 1:-1, 1:-1] = P_new
    S_new = S.clone()
    S_new[1:-1, 1:-1, 1:-1, 3] = P
    return S_new

# LES model
def compute_sgs_term(S, dx):
    grads = compute_gradients(S, dx)
    Sxx, Sxy, Sxz = grads[0], 0.5 * (grads[1] + grads[3]), 0.5 * (grads[2] + grads[6])
    Syy, Syz = grads[4], 0.5 * (grads[5] + grads[7])
    Szz = grads[8]
    S_mag = torch.sqrt(2 * (Sxx**2 + Syy**2 + Szz**2 + 2 * (Sxy**2 + Syz**2 + Sxz**2)))
    nu_t = (Cs * dx)**2 * S_mag
    nu_t_full = torch.zeros_like(S[:, :, :, 0])[1:-1, 1:-1, 1:-1]
    nu_t_full[:] = nu_t
    return nu_t_full

# Spectral smoothing
def spectral_smooth(S, N, dx):
    for i in range(3):
        vel = S[:, :, :, i]
        vel_hat = torch.fft.fftn(vel)
        k = torch.fft.fftfreq(N, d=dx, device=device) * 2 * np.pi
        kx, ky, kz = torch.meshgrid(k, k, k, indexing='ij')
        k2 = kx**2 + ky**2 + kz**2
        filter_mask = (k2 < (N * np.pi / 4)**2).float()
        vel_hat_filtered = vel_hat * filter_mask
        S[:, :, :, i] = torch.fft.ifftn(vel_hat_filtered).real
    return S

# Stability check
def check_stability_and_computability(S, t, M, dx):
    grads = compute_gradients(S, dx)
    grad_norm = torch.sqrt(sum(g**2 for g in grads[:9])).mean()
    variance = torch.var(S[:, :, :, 0:3])
    
    if grad_norm > 1e3:
        M[1:-1, 1:-1, 1:-1] = 1
        adjusted_dt = dt_min
    else:
        stability_input = torch.tensor([[grad_norm.item(), variance.item(), t]], dtype=torch.float32, device=device)
        adjustment = stability_net(stability_input)
        adjusted_dt = adjustment.item() * dt
        if adjusted_dt < dt_min:
            M[1:-1, 1:-1, 1:-1] = 1
            adjusted_dt = dt_min

    training_data.append([grad_norm.item(), variance.item(), t])
    training_labels.append(1.0 if adjusted_dt > dt_min else 0.0)
    return adjusted_dt, M, grad_norm

# Navier-Stokes step
def navier_stokes_step(S, dt, nu, rho, M, N, dx, use_dst=False):
    if M[1:-1, 1:-1, 1:-1].sum() == (N-2)**3:
        return S, M, False

    u, v, w = S[:, :, :, 0], S[:, :, :, 1], S[:, :, :, 2]
    grads = compute_gradients(S, dx)
    grad_u_x, grad_u_y, grad_u_z = grads[0:3]
    grad_v_x, grad_v_y, grad_v_z = grads[3:6]
    grad_w_x, grad_w_y, grad_w_z = grads[6:9]
    grad_P_x, grad_P_y, grad_P_z = grads[9:12]

    conv_u = u[1:-1, 1:-1, 1:-1] * grad_u_x + v[1:-1, 1:-1, 1:-1] * grad_u_y + w[1:-1, 1:-1, 1:-1] * grad_u_z
    conv_v = u[1:-1, 1:-1, 1:-1] * grad_v_x + v[1:-1, 1:-1, 1:-1] * grad_v_y + w[1:-1, 1:-1, 1:-1] * grad_v_z
    conv_w = u[1:-1, 1:-1, 1:-1] * grad_w_x + v[1:-1, 1:-1, 1:-1] * grad_w_y + w[1:-1, 1:-1, 1:-1] * grad_w_z

    nu_t = compute_sgs_term(S, dx)
    effective_nu = nu + nu_t
    lap_u = laplacian(S, 0, dx)
    lap_v = laplacian(S, 1, dx)
    lap_w = laplacian(S, 2, dx)

    u_new = u[1:-1, 1:-1, 1:-1] + dt * (-conv_u + effective_nu * lap_u)
    v_new = v[1:-1, 1:-1, 1:-1] + dt * (-conv_v + effective_nu * lap_v)
    w_new = w[1:-1, 1:-1, 1:-1] + dt * (-conv_w + effective_nu * lap_w)

    S_new = S.clone()
    S_new[1:-1, 1:-1, 1:-1, 0] = torch.where(M[1:-1, 1:-1, 1:-1] == 0, u_new, S[1:-1, 1:-1, 1:-1, 0])
    S_new[1:-1, 1:-1, 1:-1, 1] = torch.where(M[1:-1, 1:-1, 1:-1] == 0, v_new, S[1:-1, 1:-1, 1:-1, 1])
    S_new[1:-1, 1:-1, 1:-1, 2] = torch.where(M[1:-1, 1:-1, 1:-1] == 0, w_new, S[1:-1, 1:-1, 1:-1, 2])

    div = divergence(S_new, dx)
    use_dst = (div.abs().max() > divergence_threshold)
    S_new = solve_pressure_dst(S_new, rho, dt, N, dx) if use_dst else solve_pressure_fft(S_new, rho, dt, N, dx)

    _, _, _, _, _, _, _, _, _, grad_P_x, grad_P_y, grad_P_z = compute_gradients(S_new, dx)
    S_new[1:-1, 1:-1, 1:-1, 0] -= dt * grad_P_x / rho
    S_new[1:-1, 1:-1, 1:-1, 1] -= dt * grad_P_y / rho
    S_new[1:-1, 1:-1, 1:-1, 2] -= dt * grad_P_z / rho

    S_new = spectral_smooth(S_new, N, dx)
    return S_new, M, use_dst

# Subgrid refinement
def refine_subgrid(S, M, N_coarse, dx_coarse, N_fine, dx_fine, region, nu, rho):
    i_start, i_end, j_start, j_end, k_start, k_end = region
    S_region = S[i_start:i_end, j_start:j_end, k_start:k_end, :].clone()
    M_region = M[i_start:i_end, j_start:j_end, k_start:k_end].clone()

    # Check shapes
    print(f"S_region shape before permute: {S_region.shape}")
    S_region_permuted = S_region.permute(3, 0, 1, 2)  # [C, X, Y, Z]
    print(f"S_region shape after permute: {S_region_permuted.shape}")

    # Ensure cubic region for interpolation
    if S_region.shape[0] != S_region.shape[1] or S_region.shape[1] != S_region.shape[2]:
        raise ValueError(f"S_region must be cubic, got shape {S_region.shape}")

    S_fine = torch.nn.functional.interpolate(
        S_region_permuted,
        size=(N_fine, N_fine, N_fine),
        mode='trilinear',
        align_corners=True
    )
    S_fine = S_fine.permute(1, 2, 3, 0)  # Back to [X, Y, Z, C]
    M_fine = torch.zeros((N_fine, N_fine, N_fine), dtype=torch.int32, device=device)

    dt_fine = dt * (dx_fine / dx_coarse)**2
    for t in range(100):
        adjusted_dt, M_fine, _ = check_stability_and_computability(S_fine, t * dt_fine, M_fine, dx_fine)
        S_fine, M_fine, _ = navier_stokes_step(S_fine, adjusted_dt, nu, rho, M_fine, N_fine, dx_fine)
        if torch.any(torch.isnan(S_fine)):
            print("NaN detected in fine grid, reverting to coarse.")
            return S, M

    S_coarse = torch.nn.functional.interpolate(
        S_fine.permute(3, 0, 1, 2),
        size=(i_end-i_start, j_end-j_start, k_end-k_start),
        mode='trilinear',
        align_corners=True
    )
    S_coarse = S_coarse.permute(1, 2, 3, 0)
    S[i_start:i_end, j_start:j_end, k_start:k_end, :] = S_coarse
    M[i_start:i_end, j_start:j_end, k_start:k_end] = (
        M_fine[::N_fine//(i_end-i_start), ::N_fine//(j_end-j_start), ::N_fine//(k_end-k_start)] > 0
    ).int()

    return S, M

# Main simulation
reynolds_number = 10000
nu = 1.0 / reynolds_number
results = {}
divergence_history = []
energy_history = []
negative_space_history = []
time_per_step = []

print("Running coarse grid simulation...")
start_time = time.time()
for t in range(steps):
    step_start = time.time()
    adjusted_dt, M, grad_norm = check_stability_and_computability(S, t * dt, M, dx)
    S, M, use_dst = navier_stokes_step(S, adjusted_dt, nu, rho, M, N, dx)
    step_time = time.time() - step_start
    time_per_step.append(step_time)

    div = divergence(S, dx).abs().mean().item()
    energy = 0.5 * (S[:, :, :, 0:3]**2).sum().item()
    negative_space_fraction = M.sum().item() / (N**3)
    divergence_history.append(div)
    energy_history.append(energy)
    negative_space_history.append(negative_space_fraction)

    if torch.any(torch.isnan(S)):
        print(f"Simulation crashed at step {t}")
        break

    if t % 100 == 0:
        print(f"Step {t}, Div: {div:.2e}, Energy: {energy:.2f}, Negative Space: {negative_space_fraction:.2%}")

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

# Refinement
negative_space_prev = negative_space_history[-1]
for iteration in range(max_iterations):
    print(f"\nRefinement Iteration {iteration + 1}")
    N_fine = 64
    dx_fine = 1.0 / (N_fine - 1)
    if dx_fine < dx_min:
        break
    # Use cubic region for refinement
    region_size = N // 2
    region = (N//4, N//4 + region_size, N//4, N//4 + region_size, N//4, N//4 + region_size)
    S, M = refine_subgrid(S, M, N, dx, N_fine, dx_fine, region, nu, rho)

    for t in range(50):
        adjusted_dt, M, _ = check_stability_and_computability(S, t * dt, M, dx)
        S, M, _ = navier_stokes_step(S, adjusted_dt, nu, rho, M, N, dx)

    div = divergence(S, dx).abs().mean().item()
    energy = 0.5 * (S[:, :, :, 0:3]**2).sum().item()
    negative_space_fraction = M.sum().item() / (N**3)
    divergence_history.append(div)
    energy_history.append(energy)
    negative_space_history.append(negative_space_fraction)

    if negative_space_fraction >= negative_space_prev * 0.99:
        break
    negative_space_prev = negative_space_fraction

# Train StabilityNet
if training_data:
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

# Visualizations
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(divergence_history, label="Divergence")
plt.xlabel("Step")
plt.ylabel("Mean Abs Divergence")
plt.title("Divergence Over Time")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(energy_history, label="Energy")
plt.xlabel("Step")
plt.ylabel("Kinetic Energy")
plt.title("Energy Over Time")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(negative_space_history, label="Negative Space")
plt.xlabel("Step")
plt.ylabel("Fraction")
plt.title("Negative Uncomputable Space")
plt.grid(True)

plt.subplot(2, 2, 4)
u_slice = S[:, :, N//2, 0].cpu().detach().numpy()
plt.imshow(u_slice, cmap="viridis", origin="lower")
plt.colorbar(label="u-velocity")
plt.title("u-Velocity at Midplane")

plt.tight_layout()
plt.show()

# 3D Voxel Plot
M_np = M.cpu().detach().numpy()
x, y, z = np.indices(M_np.shape)
fig = go.Figure(data=go.Volume(
    x=x.flatten(), y=y.flatten(), z=z.flatten(),
    value=M_np.flatten(), isomin=0.5, isomax=1,
    opacity=0.3, surface_count=20, colorscale="Reds"
))
fig.update_layout(title="Negative Uncomputable Space (M=1 Regions)",
                 scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"))
fig.show()

print(f"Total time: {total_time:.2f}s, Avg time/step: {np.mean(time_per_step):.3f}s")

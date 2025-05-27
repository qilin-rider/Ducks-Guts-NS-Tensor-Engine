%matplotlib inline
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "colab"
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
Cs = 1.0
max_iterations = 1

# Initialize state tensor with turbulent perturbation
S = torch.zeros((N, N, N, 4), dtype=torch.float32, device=device)
S[:, :, -1, 0] = 1.0  # Lid-driven cavity
torch.manual_seed(42)
S[:, :, :, 0:3] += 0.05 * torch.randn(N, N, N, 3, device=device)
S.requires_grad = True

# Metadata tensor
M = torch.zeros((N, N, N), dtype=torch.int64, device=device)

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
    div = divergence(S, dx)[1:-1, 1:-1, 1:-1]
    P = torch.zeros_like(div)
    for _ in range(iterations):
        P_lap = (P[2:, 1:-1, 1:-1] + P[:-2, 1:-1, 1:-1] +
                 P[1:-1, 2:, 1:-1] + P[1:-1, :-2, 1:-1] +
                 P[1:-1, 1:-1, 2:] + P[1:-1, 1:-1, :-2] -
                 6 * P[1:-1, 1:-1, 1:-1]) / (dx**2)
        P_new = P[1:-1, 1:-1, 1:-1] - (dx**2) * (rho * div[1:-1, 1:-1, 1:-1] / dt

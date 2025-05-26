def divergence(S, dx):
    grad_u_x, grad_u_y, grad_u_z, grad_v_x, grad_v_y, grad_v_z, grad_w_x, grad_w_y, grad_w_z, _, _, _ = compute_gradients(S, dx)
    div = grad_u_x + grad_v_y + grad_w_z
    return div

def laplacian(S, idx, dx):
    N = S.shape[0]
    lap = torch.zeros((N, N, N), device=S.device)
    # Central differences for interior points
    lap[1:-1, 1:-1, 1:-1] = (S[2:, 1:-1, 1:-1, idx] + S[:-2, 1:-1, 1:-1, idx] +
                              S[1:-1, 2:, 1:-1, idx] + S[1:-1, :-2, 1:-1, idx] +
                              S[1:-1, 1:-1, 2:, idx] + S[1:-1, 1:-1, :-2, idx] -
                              6 * S[1:-1, 1:-1, 1:-1, idx]) / (dx**2)
    # Boundary conditions: assume zero Laplacian at boundaries (simplified)
    lap[0, :, :] = lap[1, :, :]
    lap[-1, :, :] = lap[-2, :, :]
    lap[:, 0, :] = lap[:, 1, :]
    lap[:, -1, :] = lap[:, -2, :]
    lap[:, :, 0] = lap[:, :, 1]
    lap[:, :, -1] = lap[:, :, -2]
    return lap

# FFT pressure solver (simplified)
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
    S_new[:, :, :, 3] = P
    return S_new

# Simplified Navier-Stokes step (no LES, no StabilityNet)
def navier_stokes_step(S, dt, nu, rho, N, dx):
    u, v, w, P = S[:, :, :, 0], S[:, :, :, 1], S[:, :, :, 2], S[:, :, :, 3]
    grad_u_x, grad_u_y, grad_u_z, grad_v_x, grad_v_y, grad_v_z, grad_w_x, grad_w_y, grad_w_z, grad_P_x, grad_P_y, grad_P_z = compute_gradients(S, dx)

    # Nonlinear convective terms
    conv_u = u * grad_u_x + v * grad_u_y + w * grad_u_z
    conv_v = u * grad_v_x + v * grad_v_y + w * grad_v_z
    conv_w = u * grad_w_x + v * grad_w_y + w * grad_w_z

    # Viscous terms
    lap_u = laplacian(S, 0, dx)
    lap_v = laplacian(S, 1, dx)
    lap_w = laplacian(S, 2, dx)

    # Update velocities
    u_new = u + dt * (-conv_u + nu * lap_u)
    v_new = v + dt * (-conv_v + nu * lap_v)
    w_new = w + dt * (-conv_w + nu * lap_w)

    S_new = S.clone()
    S_new[:, :, :, 0] = u_new
    S_new[:, :, :, 1] = v_new
    S_new[:, :, :, 2] = w_new

    # Solve pressure
    S_new = solve_pressure_fft(S_new, rho, dt, N, dx)

    # Correct velocities
    _, _, _, _, _, _, _, _, _, grad_P_x, grad_P_y, grad_P_z = compute_gradients(S_new, dx)
    S_new[:, :, :, 0] -= dt * grad_P_x / rho
    S_new[:, :, :, 1] -= dt * grad_P_y / rho
    S_new[:, :, :, 2] -= dt * grad_P_z / rho

    return S_new

# Main simulation
print(f"\nRunning simulation at Re={reynolds_number} on {N}^3 grid...")
for t in range(steps):
    S = navier_stokes_step(S, dt, nu, rho, N, dx)
    if t % 100 == 0:
        div = divergence(S, dx).abs().mean().item()
        print(f"Step {t}, Divergence: {div:.2e}")
    if torch.any(torch.isnan(S)):
        print(f"Simulation crashed at step {t}")
        break

# Generate the u-velocity plot at z = 16
u_slice = S[:, :, N//2, 0].cpu().detach().numpy()  # u-component at z = 16
plt.imshow(u_slice, cmap="viridis", origin="lower")
plt.colorbar(label="u-velocity")
plt.xlabel("x")
plt.ylabel("y")
plt.title(f"u-Velocity at z = {N//2}")
plt.savefig("u_velocity_z16.png")  # Save the plot
plt.show()

print("Graph generated and saved as 'u_velocity_z16.png'.")

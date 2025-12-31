import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class ReconstructionSolver:
    def __init__(self, model, operator, config):
        self.model = model
        self.operator = operator
        self.config = config
        self.device = config["device"]
        self.img_size = config["data"]["img_size"]

        # Pre-compute normalization for stability
        self.grad_norm_factor = 1.0 / (
            self.img_size * np.sqrt(self.operator.num_angles)
        )

    def get_estimated_x1(self, x_t, v_t, t):
        return x_t + (1 - t) * v_t

    def conjugate_gradient_projection(self, x, y, n_iter=5):
        """
        Solves for delta_x in: (K^T K) delta_x = K^T (y - Kx)
        using Conjugate Gradient.
        This finds the minimal update to 'x' such that K(x_new) ~ y.
        """
        # 1. Calculate the Residual in Sinogram Domain
        x_phys = (x + 1) / 2
        r_sino = y - self.operator.K(x_phys)

        # 2. Setup the Normal Equation: A * delta_x = b
        # A = K^T K
        # b = K^T r_sino
        b = self.operator.K_T(r_sino)

        # Initial guess for delta_x is 0
        delta_x = torch.zeros_like(b)

        # Initial residual for CG: r = b - A*0 = b
        r = b.clone()
        p = r.clone()
        rsold = torch.sum(r * r)

        # CG Loop
        for i in range(n_iter):
            # Ap = K^T K p
            Kp = self.operator.K(p)
            Ap = self.operator.K_T(Kp)

            # Alpha
            alpha = rsold / (torch.sum(p * Ap) + 1e-6)

            # Update delta_x
            delta_x = delta_x + alpha * p

            # Update residual
            r = r - alpha * Ap

            # Check convergence
            rsnew = torch.sum(r * r)
            if torch.sqrt(rsnew) < 1e-6:
                break

            # Update search direction
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        # 3. Apply the update
        # We need to scale delta_x back because our physics domain was [0,1]
        # but x is [-1, 1]. The derivative factor is 2.0.
        correction = delta_x * 2.0

        # x_new = x + correction
        return x + correction

    def refine(self, x_initial, steps=20, t_start=0.2):
        """
        Refinement step: Adds noise to an existing image and denoises it
        to restore high-frequency texture (personality).
        """
        self.model.eval()
        B = x_initial.shape[0]

        # 1. Add noise to go back to time t_start
        # Formula for OT-CFM: x_t = (1-t)x_0 + t*x_1
        # We assume x_initial is x_1 (approx). We sample new x_0.
        x0_new = torch.randn_like(x_initial)

        # x_t_start (The noisy starting point)
        x = (1 - t_start) * x0_new + t_start * x_initial

        # 2. Solve forward from t_start to 1
        # We use standard Euler ODE (no data consistency or very weak)
        # This allows the model to dream up texture freely.

        dt = (1.0 - t_start) / steps

        # We iterate from step corresponding to t_start
        # (Approximate loop for simplicity)
        for i in range(steps):
            # Current time t (starts at t_start and goes to 1)
            t_val = t_start + i * dt
            t = torch.full((B,), t_val, device=self.device).float()

            with torch.no_grad():
                v = self.model(x, t)
                x = x + v * dt

        return (x.clamp(-1, 1) + 1) / 2

    def data_consistency_projection(self, x, measurement):
        """
        Projects x onto the subspace defined by y = Kx.
        Returns projected x and the residual norm (loss) for tracking.
        """
        x_phys = (x + 1) / 2
        y_hat = self.operator.K(x_phys)
        residual = y_hat - measurement

        # Calculate loss for tracking
        loss_val = torch.norm(residual).item()

        correction_phys = self.operator.K_T(residual)
        correction = correction_phys * 2.0 * self.grad_norm_factor
        return x - correction, loss_val

    def solve(
        self,
        measurement,
        method="flowdps",
        steps=100,
        scale=1.0,
        noise_scale=0.0,
        save_snapshots=False,
        save_path=None,
    ):
        self.model.eval()
        B = measurement.shape[0]
        x = torch.randn(B, 1, self.img_size, self.img_size).to(self.device)
        dt = 1.0 / steps

        history = []
        loss_history = []

        # --- TQDM Progress Bar ---
        pbar = tqdm(range(steps), desc=f"Reconstructing ({method})")

        for i in pbar:
            t_scalar = i / steps
            t = torch.full((B,), t_scalar, device=self.device).float()

            current_loss = 0.0

            # --- METHOD 1: FlowDPS ---
            if method == "flowdps":
                x = x.detach().requires_grad_(True)
                v_pred = self.model(x, t)
                x1_est = self.get_estimated_x1(x, v_pred, t_scalar)

                # Loss on predicted x1
                x1_phys = (x1_est + 1) / 2
                residual = self.operator.K(x1_phys) - measurement
                loss = torch.sum(residual**2)

                grad = torch.autograd.grad(loss, x)[0]

                # Calculate Norms
                grad_norm = torch.norm(grad.flatten())
                v_norm = torch.norm(v_pred.flatten())

                # Adaptive Scaling
                if grad_norm > 1e-6:
                    adaptive_scale = scale * (v_norm / grad_norm)
                else:
                    adaptive_scale = 0

                x = x + v_pred * dt - adaptive_scale * (grad * self.grad_norm_factor)

                current_loss = loss.item()

            # --- METHOD 2: PnP-Flow ---
            elif method == "pnp":
                # Flow Step
                with torch.no_grad():
                    v_pred = self.model(x, t)
                    x = x + v_pred * dt

                # Data Consistency Step
                x = x.detach().requires_grad_(True)
                # PnP usually calculates loss on the current iterate x
                x_phys = (x + 1) / 2
                residual = self.operator.K(x_phys) - measurement
                loss = torch.sum(residual**2)

                grad = torch.autograd.grad(loss, x)[0]
                x = x - scale * (grad * self.grad_norm_factor)

                current_loss = loss.item()

            # --- METHOD 3: ICTM ---
            elif method == "ictm":
                # 1. Flow Step: Move along the prior
                with torch.no_grad():
                    v_pred = self.model(x, t)
                    x_flow = x + v_pred * (1.0 / steps)

                # 2. Projection Step (Hard Constraint)
                # Instead of a weak gradient step, we run CG to ensure Kx matches y
                # 'scale' here controls the mix: 1.0 = Full Projection, 0.0 = No Projection
                # Ideally scale=1.0 for true ICTM, but 0.5 is safer for noisy data.

                # We project x_flow onto the measurement subspace
                x_proj = self.conjugate_gradient_projection(
                    x_flow, measurement, n_iter=2
                )

                # Blend (Relaxed Projection)
                # This helps manage noise in 'y' so we don't overfit to it
                x = (1 - scale) * x_flow + scale * x_proj

                # Loss Calculation for viz
                with torch.no_grad():
                    current_loss = torch.norm(
                        self.operator.K((x + 1) / 2) - measurement
                    ).item()

            # --- METHOD 4: FLOWERS ---
            elif method == "flowers":
                x = x.detach().requires_grad_(True)
                v_pred = self.model(x, t)

                # Correction based on x1 estimate
                x1_est = self.get_estimated_x1(x, v_pred, t_scalar)
                x1_phys = (x1_est + 1) / 2

                y_hat = self.operator.K(x1_phys)
                residual = y_hat - measurement
                current_loss = torch.norm(residual).item()  # Track loss

                grad_fidelity = self.operator.K_T(residual)
                correction = grad_fidelity * 2.0 * self.grad_norm_factor

                x = x + (v_pred - scale * correction) * dt

            else:
                with torch.no_grad():
                    v = self.model(x, t)
                    x = x + v * dt

            # Add Stochastic Injection
            if noise_scale > 0:
                # We inject noise proportional to step size
                noise = torch.randn_like(x) * np.sqrt(dt) * noise_scale
                x = x + noise

            # Safety Clamp & Snapshot
            x = x.detach().clamp(-3.0, 3.0)

            # Log Loss
            loss_history.append(current_loss)
            pbar.set_postfix({"Loss": f"{current_loss:.2e}"})

            if save_snapshots and i % (steps // 10) == 0:
                history.append(x.cpu())

        # --- Plotting ---
        if save_path:
            plt.figure(figsize=(10, 5))
            plt.plot(loss_history, label="Data Consistency Loss")
            plt.yscale("log")  # Log scale is better for convergence plots
            plt.xlabel("Iteration")
            plt.ylabel("Loss (Log Scale)")
            plt.title(f"Convergence - {method}")
            plt.grid(True, which="both", ls="-", alpha=0.5)
            plt.legend()

            # Save inside the results folder
            plot_file = os.path.join(save_path, "loss_curve.png")
            plt.savefig(plot_file)
            plt.close()

        return (x.clamp(-1, 1) + 1) / 2, history

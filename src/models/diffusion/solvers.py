import numpy as np
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm


class DiffusionSolver:
    def __init__(self, model, device="cuda", num_train_timesteps=1000):
        self.model = model
        self.device = device
        self.num_train_timesteps = num_train_timesteps

        # Standard Linear DDPM Schedule
        self.beta_start = 0.0001
        self.beta_end = 0.02
        self.betas = torch.linspace(
            self.beta_start, self.beta_end, num_train_timesteps
        ).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def _extract(self, a, t, x_shape):
        out = a.gather(-1, t)
        return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1))).to(self.device)

    def get_timesteps_seq(self, num_inference_steps):
        steps = np.linspace(
            0, self.num_train_timesteps - 1, num_inference_steps, dtype=int
        )
        return list(reversed(steps))

    def predict_x0_from_epsilon(self, x_t, t, epsilon):
        """
        Tweedy's Formula: Estimate x_0 from x_t and epsilon.
        """
        sqrt_alpha_bar = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_bar = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        return (x_t - sqrt_one_minus_alpha_bar * epsilon) / sqrt_alpha_bar

    def solve_cgls(self, operator, y, x_init, n_iter=5):
        """CGLS solver for DDRM data consistency."""
        x = x_init.clone()
        r = y - operator.K(x)
        p = operator.K_T(r)
        s = p.clone()
        gamma = torch.sum(s.view(s.shape[0], -1) ** 2, dim=1, keepdim=True).view(
            s.shape[0], 1, 1, 1
        )

        for _ in range(n_iter):
            q = operator.K(p)
            norm_q = torch.sum(q.view(q.shape[0], -1) ** 2, dim=1, keepdim=True).view(
                q.shape[0], 1, 1, 1
            )
            alpha = gamma / (norm_q + 1e-6)
            x = x + alpha * p
            r = r - alpha * q
            s = operator.K_T(r)
            gamma_new = torch.sum(
                s.view(s.shape[0], -1) ** 2, dim=1, keepdim=True
            ).view(s.shape[0], 1, 1, 1)
            beta = gamma_new / (gamma + 1e-6)
            p = s + beta * p
            gamma = gamma_new
        return x

    def _compute_metrics(self, x_pred, x_ref):
        """Helper to compute metrics efficiently for pbar."""
        if x_ref is None:
            return {}

        # Ensure range [0, 1] for torchmetrics
        x_p = x_pred.clamp(0, 1)
        x_r = x_ref.clamp(0, 1)

        psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)(x_p, x_r)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)(
            x_p, x_r
        )

        return {"psnr": f"{psnr.item():.2f}", "ssim": f"{ssim.item():.4f}"}

    def ddim_inversion(self, x0_image, num_inference_steps=50):
        """
        Inverts a clean image (x0) back to latent noise (xT) using DDIM ODE.
        x0_image: Tensor in range [-1, 1]
        """
        # Create Forward Sequence (0 -> T)
        timesteps = list(reversed(self.get_timesteps_seq(num_inference_steps)))

        # We need pairs (t, t_next) where t_next > t
        timesteps_next = timesteps[1:] + [
            self.num_train_timesteps - 1
        ]  # Clamp last to 999

        x = x0_image.clone()
        batch_size = x.shape[0]

        pbar = tqdm(
            zip(timesteps, timesteps_next), total=len(timesteps), desc="DDIM Inversion"
        )

        for t_cur, t_next in pbar:
            t_tensor = torch.full(
                (batch_size,), t_cur, device=self.device, dtype=torch.long
            )

            with torch.no_grad():
                epsilon = self.model(x, t_tensor)

            # 1. Get coefficients for current t
            alpha_bar_t = self.alphas_cumprod[t_cur]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)

            # 2. Get coefficients for next t (t+1)
            alpha_bar_next = self.alphas_cumprod[t_next]
            sqrt_alpha_bar_next = torch.sqrt(alpha_bar_next)
            sqrt_one_minus_alpha_bar_next = torch.sqrt(1 - alpha_bar_next)

            # 3. Estimate x0 (Reversal of forward eq)
            # x_t = sqrt(alpha)*x0 + sqrt(1-alpha)*eps
            # x0 = (x_t - sqrt(1-alpha)*eps) / sqrt(alpha)
            x0_hat = (x - sqrt_one_minus_alpha_bar_t * epsilon) / sqrt_alpha_bar_t

            # 4. Deterministic Forward Step to t_next
            # x_{t+1} = sqrt(alpha_next)*x0_hat + sqrt(1-alpha_next)*epsilon
            x = sqrt_alpha_bar_next * x0_hat + sqrt_one_minus_alpha_bar_next * epsilon

        return x

    # =========================================================
    # Algorithm: DPS (Supports x_init)
    # =========================================================
    def sample_dps(
        self,
        measurement,
        operator,
        image_size,
        noise_scale=0.0,
        zeta=1.0,
        num_inference_steps=50,
        clean_ref=None,
        x_init=None,
    ):
        """
        DPS Sampling.
        x_init: (Optional) Starting latent tensor (e.g. Inverted FBP).
                If None, starts from standard Gaussian noise.
        """
        batch_size = measurement.shape[0]
        shape = (batch_size, 1, image_size, image_size)

        if x_init is not None:
            print("[INFO] Initializing DPS from provided latent (FBP Inversion).")
            x = x_init.clone()
        else:
            x = torch.randn(shape, device=self.device)

        timesteps = self.get_timesteps_seq(num_inference_steps)
        timesteps_next = timesteps[1:] + [-1]

        pbar = tqdm(timesteps, desc=f"DPS")

        for i, t_cur in enumerate(pbar):
            t_prev = timesteps_next[i]
            t_tensor = torch.full(
                (batch_size,), t_cur, device=self.device, dtype=torch.long
            )

            x = x.detach().requires_grad_()

            # Predict
            epsilon = self.model(x, t_tensor)
            x_0_hat = self.predict_x0_from_epsilon(x, t_tensor, epsilon)

            # Physics Guidance
            x_0_phys = (x_0_hat + 1) / 2
            residual = operator.K(x_0_phys) - measurement
            loss = torch.sum(residual**2)

            grad = torch.autograd.grad(loss, x)[0]

            # Calculate Norms
            grad_norm = torch.norm(grad.flatten())
            eps_norm = torch.norm(epsilon.flatten())

            # Adaptive Scaling
            if grad_norm > 1e-6:
                adaptive_scale = zeta * (eps_norm / grad_norm)
            else:
                adaptive_scale = 0

            # DDIM Update
            x = x.detach()
            if t_prev < 0:
                alpha_bar_prev = torch.tensor(1.0, device=self.device)
            else:
                alpha_bar_prev = self.alphas_cumprod[t_prev]

            sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
            sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)

            x_prev = (
                sqrt_alpha_bar_prev * x_0_hat + sqrt_one_minus_alpha_bar_prev * epsilon
            )

            # Apply Guidance
            x = x_prev - adaptive_scale * grad

            # Add Stochastic Injection
            if noise_scale > 0:
                # We inject noise proportional to step size
                noise = (
                    torch.randn_like(x) * sqrt_one_minus_alpha_bar_prev * noise_scale
                )
                x = x + noise

            # Safety Clamp
            x = x.clamp(-3.0, 3.0)

            # Metrics
            x_0_final = (x_0_hat.clamp(-1, 1) + 1.0) / 2.0
            metrics = self._compute_metrics(x_0_final.detach(), clean_ref)
            if metrics:
                pbar.set_postfix(metrics)

        return (x.clamp(-1, 1) + 1.0) / 2.0

    # =========================================================
    # Algorithm 2: DDRM (Unchanged, works well)
    # =========================================================
    def sample_ddrm(
        self,
        measurement,
        operator,
        image_size,
        scale_factor=1.0,
        num_inference_steps=50,
        cgls_steps=5,
        clean_ref=None,
        x_init=None,
    ):
        batch_size = measurement.shape[0]
        shape = (batch_size, 1, image_size, image_size)

        if x_init is not None:
            print("[INFO] Initializing DPS from provided latent (FBP Inversion).")
            x = x_init.clone()
        else:
            x = torch.randn(shape, device=self.device)

        timesteps = self.get_timesteps_seq(num_inference_steps)
        timesteps_next = timesteps[1:] + [-1]

        pbar = tqdm(timesteps, desc=f"DDRM")

        for i, t_cur in enumerate(pbar):
            t_prev = timesteps_next[i]
            t_tensor = torch.full(
                (batch_size,), t_cur, device=self.device, dtype=torch.long
            )

            with torch.no_grad():
                epsilon = self.model(x, t_tensor)
                x_0_hat = self.predict_x0_from_epsilon(x, t_tensor, epsilon)

            # Physics Correction
            x_0_phys = (x_0_hat + 1.0) / 2.0
            y_phys = measurement * scale_factor

            x_0_phys_corrected = self.solve_cgls(
                operator, y_phys, x_0_phys, n_iter=cgls_steps
            )
            x_0_hat_corrected = (x_0_phys_corrected * 2.0) - 1.0

            metrics = self._compute_metrics(x_0_phys_corrected.detach(), clean_ref)
            if metrics:
                pbar.set_postfix(metrics)

            if t_prev < 0:
                x = x_0_hat_corrected
            else:
                alpha_bar_prev = self.alphas_cumprod[t_prev]
                sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
                sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)

                x = (
                    sqrt_alpha_bar_prev * x_0_hat_corrected
                    + sqrt_one_minus_alpha_bar_prev * epsilon
                )

        return (x.clamp(-1, 1) + 1.0) / 2.0

import astra
import numpy as np
import torch


class AstraProjectorFunction(torch.autograd.Function):
    """
    This class bridges the gap between PyTorch (Tensors) and ASTRA (Numpy).
    It defines how to compute the Forward pass (Projection)
    and the Backward pass (Backprojection) so autograd works.
    """

    @staticmethod
    def forward(ctx, x, operator):
        # Save the operator context for the backward pass
        ctx.operator = operator

        # Call the internal numpy implementation
        # We can safely detach here because we are inside a custom Autograd Function
        return operator._compute_forward(x)

    @staticmethod
    def backward(ctx, grad_output):
        # The gradient of the Projection is the Backprojection.
        # grad_output represents dL/dy (gradient at the sinogram)
        operator = ctx.operator

        # We compute dL/dx = K^T * dL/dy
        grad_input = operator._compute_backward(grad_output)

        # Return gradient for x, and None for the operator argument
        return grad_input, None


class CTOperator:
    def __init__(self, img_size, num_angles, detector_size, device="cuda"):
        self.img_size = img_size
        self.num_angles = num_angles
        self.detector_size = detector_size
        self.device = device

        # Geometry Setup
        self.angles = np.linspace(0, np.pi, num_angles, endpoint=False)
        self.vol_geom = astra.create_vol_geom(img_size, img_size)
        self.proj_geom = astra.create_proj_geom(
            "parallel", 1.0, detector_size, self.angles
        )

        # Create Projector ID (using CUDA)
        self.proj_id = astra.create_projector("cuda", self.proj_geom, self.vol_geom)

    def _compute_forward(self, x):
        """Internal Helper: Numpy/ASTRA implementation of K(x)"""
        B = x.shape[0]
        y_list = []
        # Detach to numpy for ASTRA
        x_np = x.detach().cpu().numpy()

        for i in range(B):
            img = np.ascontiguousarray(x_np[i, 0], dtype=np.float32)
            id_sino, sino = astra.create_sino(img, self.proj_id)
            y_list.append(sino)
            astra.data2d.delete(id_sino)

        y = np.stack(y_list)
        return torch.from_numpy(y).unsqueeze(1).to(self.device).float()

    def _compute_backward(self, y):
        """Internal Helper: Numpy/ASTRA implementation of K^T(y)"""
        B = y.shape[0]
        x_list = []
        # Detach to numpy for ASTRA
        y_np = y.detach().cpu().numpy()

        for i in range(B):
            sino = np.ascontiguousarray(y_np[i, 0], dtype=np.float32)
            id_bp, bp = astra.create_backprojection(sino, self.proj_id)
            x_list.append(bp)
            astra.data2d.delete(id_bp)

        x = np.stack(x_list)
        return torch.from_numpy(x).unsqueeze(1).to(self.device).float()

    def K(self, x):
        """
        Differentiable Forward Projection (Image -> Sinogram).
        Uses the custom autograd function to maintain the gradient chain.
        """
        return AstraProjectorFunction.apply(x, self)

    def K_T(self, y):
        """
        Back Projection (Sinogram -> Image).
        Standard adjoint operation.
        """
        return self._compute_backward(y)

    def fbp(self, sinogram):
        """
        Computes FBP (Filtered Backprojection) using ASTRA.
        Args:
            sinogram: Tensor (B, 1, Angles, Detectors)
        Returns:
            reconstruction: Tensor (B, 1, H, W) in [0, 1] range (approx)
        """
        B = sinogram.shape[0]
        rec_list = []
        sino_np = sinogram.detach().cpu().numpy()

        for i in range(B):
            sino = np.ascontiguousarray(sino_np[i, 0], dtype=np.float32)

            # Create Sinogram ID
            sino_id = astra.data2d.create("-sino", self.proj_geom, sino)

            # Create Reconstruction Target ID
            rec_id = astra.data2d.create("-vol", self.vol_geom)

            # Setup FBP Configuration
            cfg = astra.astra_dict("FBP_CUDA")
            cfg["ReconstructionDataId"] = rec_id
            cfg["ProjectionDataId"] = sino_id
            cfg["ProjectorId"] = self.proj_id

            # Run Algorithm
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)

            # Get Result
            rec = astra.data2d.get(rec_id)
            rec_list.append(rec)

            # Cleanup
            astra.algorithm.delete(alg_id)
            astra.data2d.delete(sino_id)
            astra.data2d.delete(rec_id)

        rec_stack = np.stack(rec_list)
        return torch.from_numpy(rec_stack).unsqueeze(1).to(self.device).float()

    def cleanup(self):
        astra.projector.delete(self.proj_id)

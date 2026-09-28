import torch
import torch.nn.functional as F

from . import configuration as cfg

H = W = 64
C = 3


def gaussian_center_mask(device="cpu") -> torch.Tensor:
    """Centered Gaussian mask (H, W), normalized so that mean=1.

    Weights the central pixels of the patch more heavily, since that is
    where the cell is located.
    """
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing="ij",
    )
    w = torch.exp(-(xx ** 2 + yy ** 2) / (2 * cfg.GAUSSIAN_SIGMA ** 2))
    return w / w.mean()


def sparsity_metrics(z: torch.Tensor):
    """Return (zero_fraction, mean_active_atoms_per_sample)."""
    with torch.no_grad():
        mask   = (z == 0)
        frac_z = mask.float().mean().item()
        active = (~mask).float().sum(dim=1).mean().item()
    return frac_z, active


def mse_loss(x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor,
             lmbd_l1: float = 1.0, lmbd_rec: float = 1.0, data_range: float = None):
    B      = x.size(0)
    w      = gaussian_center_mask(device=x.device).view(1, 1, H, W).expand(B, C, H, W)
    diff   = (x_hat.view(B, C, H, W) - x.view(B, C, H, W)) ** 2
    rec    = (w * diff).mean()
    sparse = z.abs().mean()
    return lmbd_rec * rec + lmbd_l1 * sparse, rec, sparse


def mae_loss(x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor,
             lmbd_l1: float = 1.0, lmbd_rec: float = 1.0, data_range: float = None):
    B      = x.size(0)
    w      = gaussian_center_mask(device=x.device).view(1, 1, H, W).expand(B, C, H, W)
    diff   = (x_hat.view(B, C, H, W) - x.view(B, C, H, W)).abs()
    rec    = (w * diff).mean()
    sparse = z.abs().mean()
    return lmbd_rec * rec + lmbd_l1 * sparse, rec, sparse


def _ssim_kernel(kernel_size: int, sigma: float,
                 channels: int, device) -> torch.Tensor:
    # Normalized 1-D Gaussian kernel -> outer product -> separable 2-D kernel
    # Final shape: (channels, 1, kernel_size, kernel_size) for depthwise conv2d
    ksize_half = (kernel_size - 1) * 0.5
    coords = torch.linspace(-ksize_half, ksize_half, steps=kernel_size, device=device)
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    return g.outer(g).unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)


def _ssim_map(x4d: torch.Tensor, y4d: torch.Tensor,
              kernel: torch.Tensor, pad: int,
              C1: float, C2: float) -> torch.Tensor:
    """Local SSIM map with reflect padding, equivalent to ignite.metrics.SSIM."""
    ch     = x4d.shape[1]
    p      = [pad, pad, pad, pad]
    xp     = F.pad(x4d, p, mode="reflect")
    yp     = F.pad(y4d, p, mode="reflect")
    mu_x   = F.conv2d(xp,      kernel, groups=ch)
    mu_y   = F.conv2d(yp,      kernel, groups=ch)
    mu_x2  = mu_x ** 2
    mu_y2  = mu_y ** 2
    mu_xy  = mu_x * mu_y
    sig_x2 = F.conv2d(xp * xp, kernel, groups=ch) - mu_x2
    sig_y2 = F.conv2d(yp * yp, kernel, groups=ch) - mu_y2
    sig_xy = F.conv2d(xp * yp, kernel, groups=ch) - mu_xy
    num    = (2 * mu_xy + C1) * (2 * sig_xy + C2)
    denom  = (mu_x2 + mu_y2 + C1) * (sig_x2 + sig_y2 + C2)
    return num / (denom + 1e-8)


def ssim_loss(x: torch.Tensor, x_hat: torch.Tensor, z: torch.Tensor,
              lmbd_l1: float = 1.0, lmbd_rec: float = 1.0, data_range: float = None):
    if data_range is None:
        raise ValueError(
            "ssim_loss requiere data_range. "
            "Asegúrate de que hparams['ssim_data_range'] se pasa correctamente desde train.py."
        )

    B    = x.size(0)
    x4d  = x.view(B, C, H, W)
    xh4d = x_hat.view(B, C, H, W)

    kernel = _ssim_kernel(cfg.SSIM_KERNEL_SIZE, cfg.SSIM_SIGMA, C, x.device)
    pad    = cfg.SSIM_KERNEL_SIZE // 2
    C1 = (cfg.SSIM_K1 * data_range) ** 2
    C2 = (cfg.SSIM_K2 * data_range) ** 2

    ssim_map = _ssim_map(x4d, xh4d, kernel, pad, C1, C2)
    w        = gaussian_center_mask(device=x.device).view(1, 1, H, W)
    rec    = 1.0 - (ssim_map * w).mean()
    sparse = z.abs().mean()
    return lmbd_rec * rec + lmbd_l1 * sparse, rec, sparse

import torch


def apply_missing_features(x, missing_ratio, seed):
    """Mask features with given ratio and seed, fill masked entries with NaN."""
    if x.ndim != 2:
        raise ValueError(f"x must be 2D, got shape {tuple(x.shape)}")
    if not torch.is_floating_point(x):
        raise TypeError("x must be floating point")

    num_nodes, num_features = x.shape
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    missing_mask = torch.rand(
        num_nodes, num_features, generator=generator, device="cpu"
    ) < missing_ratio
    missing_mask = missing_mask.to(device=x.device)

    x_masked = x.clone()
    x_masked[missing_mask] = float("nan")
    return x_masked


def impute_missing_features(x, method='zero'):
    """Fill NaN feature entries using the selected imputation method."""
    method = method.lower()
    if method != 'zero':
        raise ValueError(f"Unsupported imputation method: {method!r}")

    x_filled = x.clone()
    x_filled[torch.isnan(x_filled)] = 0.0
    return x_filled

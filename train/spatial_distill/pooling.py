from typing import List, Tuple

import torch


def pool_mse_preds_by_sample(
    packed_mse_preds: torch.Tensor,
    mse_loss_indexes: torch.Tensor,
    sample_lens: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if packed_mse_preds is None:
        raise ValueError("packed_mse_preds is required for spatial distillation.")
    if mse_loss_indexes is None:
        raise ValueError("mse_loss_indexes is required for spatial distillation.")

    if len(sample_lens) == 0:
        return packed_mse_preds.new_zeros((0, packed_mse_preds.shape[-1])), torch.zeros(
            0, dtype=torch.bool, device=packed_mse_preds.device
        )
    if packed_mse_preds.shape[0] == 0:
        return packed_mse_preds.new_zeros((len(sample_lens), packed_mse_preds.shape[-1])), torch.zeros(
            len(sample_lens), dtype=torch.bool, device=packed_mse_preds.device
        )

    mse_loss_indexes = mse_loss_indexes.to(torch.long)
    offsets = [0]
    for n in sample_lens:
        offsets.append(offsets[-1] + int(n))

    pooled = []
    valid = []
    for i in range(len(sample_lens)):
        start = offsets[i]
        end = offsets[i + 1]
        token_mask = (mse_loss_indexes >= start) & (mse_loss_indexes < end)
        if token_mask.any():
            pooled.append(packed_mse_preds[token_mask].mean(dim=0))
            valid.append(True)
        else:
            pooled.append(torch.zeros_like(packed_mse_preds[0]))
            valid.append(False)

    pooled = torch.stack(pooled, dim=0)
    valid = torch.tensor(valid, device=packed_mse_preds.device, dtype=torch.bool)
    return pooled, valid

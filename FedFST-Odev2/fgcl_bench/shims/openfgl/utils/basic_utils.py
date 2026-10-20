# POWER'ın flcore/Ours.py'si OpenFGL'den yalnızca bu fonksiyonu import ediyor.
# Kaynak: https://github.com/xkLi-Allen/OpenFGL (commit 5483768), openfgl/utils/basic_utils.py (aynen kopya).
import torch


def idx_to_mask_tensor(idx_list, length):
    """
    Converts a list of indices to a tensor mask of a specified length.

    Args:
        idx_list (list[int]): List of indices that should be marked as 1 in the mask.
        length (int): Total length of the mask tensor.

    Returns:
        torch.Tensor: A binary mask tensor where positions corresponding to indices in idx_list are set to 1.
    """
    mask = torch.zeros(length)
    mask[idx_list] = 1
    return mask

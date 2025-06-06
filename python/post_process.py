import numpy as np

def remove_small_masks(mask, param, bound):
    W, H = mask.shape[1], mask.shape[0]

    new_mask = np.zeros_like(mask, dtype=np.int8)
    new_param = []

    for i in range(len(param)):
        mask_i = (mask == i + 1)
        if np.sum(mask_i) < bound * W * H:
            continue
        new_mask[mask_i] = new_mask.max() + 1
        new_param.append(param[i])

    return new_mask, np.array(new_param)
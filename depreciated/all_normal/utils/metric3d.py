import torch
import numpy as np

def metric3d(model, rgb):
    rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    rgb = rgb.cuda() if torch.cuda.is_available() else rgb

    pred_depth, confidence, output_dict = model.inference({'input': rgb})
    pred_normal = output_dict['prediction_normal'][:, :3, :, :] # only available for Metric3Dv2 i.e., ViT models
    normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details

    output_depth = pred_depth[0, 0].cpu().numpy()[2:-2,2:-2]
    output_normal = pred_normal[0,:].cpu().numpy().transpose(1,2,0)[2:-2,2:-2]

    return output_depth, output_normal
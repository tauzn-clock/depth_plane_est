import torch
from PIL import Image
import numpy as np
from skimage.transform import resize

def run_metric3d(rgb):
    rgb = torch.from_numpy(np.array(rgb)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    rgb = rgb.cuda() if torch.cuda.is_available() else rgb

    model = torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small', pretrain=True).cuda() 
    model = model.cuda() if torch.cuda.is_available() else model
    pred_depth, confidence, output_dict = model.inference({'input': rgb})
    pred_normal = output_dict['prediction_normal'][:, :3, :, :] # only available for Metric3Dv2 i.e., ViT models
    normal_confidence = output_dict['prediction_normal'][:, 3, :, :] # see https://arxiv.org/abs/2109.09881 for details

    # Visualize depth
    W, H = rgb.shape[3], rgb.shape[2]
    output_depth = pred_depth[0, 0].cpu().numpy()
    output_depth = resize(output_depth, (H, W), anti_aliasing=True)
    output_normal = pred_normal[0,:].cpu().numpy().transpose(1,2,0)
    output_normal = resize(output_normal, (H, W), anti_aliasing=True)
    
    return output_depth, output_normal


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rgb = Image.open('/scratchdata/stair3/rgb/0.png').convert('RGB')
    output_depth, output_normal = run_metric3d(rgb)

    plt.imsave('depth.png', output_depth)
    plt.imsave('normal.png', (output_normal+1)/2)
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from PIL import ImageFilter

from SRCNN_method.models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr
torch.cuda.empty_cache()

def get_srcnn(image_file_dir, scale):
    
    if scale==2:
        weights_file_dir='SRCNN_method/train_results/400epo_x2.pth'
    elif scale==3:
        weights_file_dir='SRCNN_method/train_results/400epo_x3.pth'
    elif scale==4:
        weights_file_dir='SRCNN_method/train_results/400epo_x4.pth'
    
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    image = pil_image.open(image_file_dir).convert('RGB')
    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file_dir, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    
    image = image.resize((image.width * scale, image.height * scale), resample=pil_image.Resampling.BICUBIC)

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)

    return output

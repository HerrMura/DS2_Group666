import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from SRCNN_method.SRCNN_app import get_srcnn
from model import SROP
from ssim import SSIM
from offset_map import Offs_Pro
from fusion import img_fusion
from utils import tensor2img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, required=True)
    args = parser.parse_args()
    if args.scale==2:
        weights_file='SROP_method/train_results/SROP_x2.pth'
    elif args.scale==3:
        weights_file='SROP_method/train_results/SROP_x3.pth'
    elif args.scale==4:
        weights_file='SROP_method/train_results/SROP_x4.pth'
    
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model=SROP(num_channels=3,scale=args.scale).to(device)
    criterion=SSIM(channels=3).to(device)
    Mapping_Offset=Offs_Pro().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    with torch.no_grad():
        image = pil_image.open(args.image_file).convert('RGB') 
        hr_width = (image.width // args.scale) * args.scale
        hr_height = (image.height // args.scale) * args.scale
        hr = image.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr_width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        
        sr_flat = lr.resize((hr_width, hr_height), resample=pil_image.LANCZOS)
        #sr_flat=get_srcnn(image, image_file_dir, scale, save=False)
        #sr_flat=np.array(sr_flat).astype(np.float32)
        sr_flat=np.array(sr_flat).astype(np.float32)
            # sr_flat is a nparray of size (sh,sw,3)
        sr_flat=torch.from_numpy(sr_flat).unsqueeze(0).permute(0,3,1,2)
            # sr_flat is a tensor of size (1,3,sh,sw) 
        lr_image = np.array(lr).astype(np.float32)
            # lr_image is a nparray of size (h,w,3)
        lr_image=torch.from_numpy(lr_image).unsqueeze(0).permute(0,3,1,2)
            # lr_image is a tensor of size (1,3,h,w)             
        sr_flat=sr_flat.to(device)
        lr_image=lr_image.to(device)
        offset_map = model(lr_image)
            # offset_map is of size (1,2,sh,sw) 
        sr_offset=Mapping_Offset(lr_image,offset_map,scale=args.scale,Train=False)
        #img_offset.show()
            # sr_offset is of size (1,3,sh,sw)
        sr_result=img_fusion(sr_flat, sr_offset, offset_map)
            # sr_result is of size (1,3,sh,sw)
        hr=np.array(hr).astype(np.float32)
        hr=torch.from_numpy(hr).unsqueeze(0).permute(0,3,1,2)
        hr=hr.to(device)
        ssim_val=criterion(hr,sr_result,as_loss=False)
        print(ssim_val)
        sr_img=tensor2img(sr_result)
        
    sr_img.save(args.image_file.replace('.', '_SROP_x{}.'.format(args.scale)))
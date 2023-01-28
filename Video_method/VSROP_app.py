import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
from SROP_method.model import SROP
from SROP_method.offset_map import Offs_Pro
from SROP_method.fusion import vid_fusion
from utils import tensor2img

def srop_vid(frame_batch,sr_flat,scale):
    # frame batch is a tensor of size (batch_size,channels,h,w)
    # sr_flat is a tensor of size (batch_size,channels,sh,sw)
    if scale==2:
        weights_file='SROP_method/train_results/SROP_x2.pth'
    elif scale==3:
        weights_file='SROP_method/train_results/SROP_x3.pth'
    elif scale==4:
        weights_file='SROP_method/train_results/SROP_x4.pth'
        
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    
    model=SROP(num_channels=3,scale=scale).to(device)
    Mapping_Offset=Offs_Pro().to(device)
    
    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    with torch.no_grad():
        lr_width=frame_batch.shape[3]
        lr_height=frame_batch.shape[2]
        hr_width=lr_width * scale
        hr_height=lr_height * scale
        offset_map = model(frame_batch).clamp(-1,1)
            # offset_map is a tensor of size (batch_size,2,sh,sw) 
        sr_offset=Mapping_Offset(frame_batch,offset_map,scale,Train=False)
            # sr_offset is a tensor of size (batch_size,3,sh,sw)
        frame_batch=frame_batch.to("cpu")
        sr_result=vid_fusion(sr_flat, sr_offset, offset_map)
            # sr_result is of size (batch,3,sh,sw)
        
    return sr_result

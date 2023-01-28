import torch.nn.functional as F
import torch
import numpy as np
# fusion the flat area and the edges


def fspecial_gauss(size, sigma, channels):
    # Function to mimic the 'fspecial' gaussian MATLAB function
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    g = torch.from_numpy(g/g.sum()).float().unsqueeze(0).unsqueeze(0)
    return g.repeat(channels,1,1,1)

def gaussian_filter(input, win):
    out = F.conv2d(input, win, stride=1, padding=1, groups=input.shape[1])
    return out

def img_fusion(sr_flat, sr_offset, offset_map):
    offset_map=torch.round(offset_map)
    offset_map=torch.abs(offset_map)
    indic_func=offset_map[0,0,:,:]+offset_map[0,1,:,:]
    indic_func=indic_func.clamp(0,1).repeat(3,1,1,1).permute(1,0,2,3)
    win=fspecial_gauss(3,1.5,3).to(sr_flat.device)
    alpha=gaussian_filter(indic_func,win)
    sr_result=alpha*sr_offset+(1-alpha)*sr_flat
    return sr_result

def vid_fusion(sr_flat, sr_offset, offset_map):
    offset_map=torch.round(offset_map)
    offset_map=torch.abs(offset_map)
    indic_func=offset_map[:,0,:,:]+offset_map[:,1,:,:]
    indic_func=indic_func.clamp(0,1).repeat(3,1,1,1).permute(1,0,2,3)
    offset_map=offset_map.to("cpu")
    win=fspecial_gauss(3,1.5,3).to(sr_flat.device)
    alpha=gaussian_filter(indic_func,win)
    sr_result=alpha*sr_offset+(1-alpha)*sr_flat
    return sr_result
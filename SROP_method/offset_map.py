import torch
import torch.nn.functional as F

def offset_projection(img, offset_map, scale, Train):
    if Train==True:
        UpSample=torch.nn.UpsamplingBilinear2d(scale_factor=scale)
    else:
        UpSample=torch.nn.UpsamplingNearest2d(scale_factor=scale)
    sr_input=UpSample(img)
    height=sr_input.shape[2]
    width=sr_input.shape[3]
    
    xy=torch.ones_like(offset_map)
    #tmp=offset_map[:,0,:,:]
    #offset_map[:,0,:,:]=offset_map[:,1,:,:]
    #offset_map[:,1,:,:]=tmp
    
    a=torch.arange(0,width,1)
    b=torch.ones(height,width)
    ab=a*b # max=width-1
    xy[:,0,0:height,0:width]=ab

    c=torch.arange(0,height,1).reshape((height,1))
    d=torch.ones(height,width)
    cd=c*d # max=height-1
    xy[:,1,0:height,0:width]=cd
    
    xy[:,0,:,:]=xy[:,0,:,:]+offset_map[:,0,:,:]
    xy[:,1,:,:]=xy[:,1,:,:]-offset_map[:,1,:,:]
    xy[:,0,:,:]=2*xy[:,0,:,:] / (width-1) - 1
    xy[:,1,:,:]=2*xy[:,1,:,:] / (height-1) - 1
    xy=xy.permute(0,2,3,1)
    if Train==True:
        sr_output=F.grid_sample(sr_input,xy,mode='bilinear',padding_mode='border',align_corners=True)
    else:
        sr_output=F.grid_sample(sr_input,xy,mode='nearest',padding_mode='border',align_corners=True)
    
    return sr_output
                
class Offs_Pro(torch.nn.Module):
    def __init__(self):
        super(Offs_Pro, self).__init__()
        
    def forward(self, img, offset_map, scale, Train):
        output=offset_projection(img,offset_map, scale, Train)
        return output
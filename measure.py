import torch
import torch.nn.functional as F
import numpy as np
import PIL.Image as pil_image

def fspecial_gauss(size, sigma, channels):
    # Function to mimic the 'fspecial' gaussian MATLAB function
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    g = torch.from_numpy(g/g.sum()).float().unsqueeze(0).unsqueeze(0)
    return g.repeat(channels,1,1,1)

def gaussian_filter(input, win):
    out = F.conv2d(input, win, stride=1, padding=0, groups=input.shape[1])
        # conv2d requires the input as tensor([batch_size,channels,height,width])
    return out

def ssim(X, Y, win, get_ssim_map=False, get_cs=False, get_weight=False):
    C1 = (0.01*255) ** 2
    C2 = (0.03*255) ** 2

    win = win.to(X.device)

    mu1 = gaussian_filter(X, win)
    mu2 = gaussian_filter(Y, win)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(X * X, win) - mu1_sq
    sigma2_sq = gaussian_filter(Y * Y, win) - mu2_sq
    sigma12 = gaussian_filter(X * Y, win) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    cs_map = F.relu(cs_map)  # force the ssim response to be nonnegative to avoid negative results.
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map
    #print(ssim_map.shape)
    ssim_val = ssim_map.mean([1, 2, 3])
    #print(ssim_val.shape)

    if get_weight:
        weights = torch.log((1 + sigma1_sq / C2) * (1 + sigma2_sq / C2))
        return ssim_map, weights

    if get_ssim_map:
        return ssim_map

    if get_cs:
        return ssim_val, cs_map.mean([1, 2, 3])

    return ssim_val


class SSIM(torch.nn.Module):
    def __init__(self, channels=3):

        super(SSIM, self).__init__()
        self.win = fspecial_gauss(11, 1.5, channels)

    def forward(self, X, Y, as_loss=True):
        assert X.shape == Y.shape
        if as_loss:
            score = ssim(X, Y, win=self.win)
            return 1 - score.mean()
        else:
            with torch.no_grad():
                score = ssim(X, Y, win=self.win)
            return score

def calc_psnr(img1, img2):
    return 10. * torch.log10(255 ** 2 / torch.mean((img1 - img2) ** 2))

def compare(img1,img2):
    array1 = np.array(img1).astype(np.float32)
    array1 = array1.swapaxes(1,2)
    array1 = array1.swapaxes(0,1)
    
    array2 = np.array(img2).astype(np.float32)
    array2 = array2.swapaxes(1,2)
    array2 = array2.swapaxes(0,1)
    
    array1=torch.from_numpy(array1).unsqueeze(0)
    array2=torch.from_numpy(array2).unsqueeze(0)
    
    calc_ssim=SSIM(channels=3)
    m1=calc_psnr(array1,array2)
    m2=calc_ssim(array1,array2,as_loss=False)
    
    print("psnr="+str(m1.item()))
    print("ssim="+str(m2.item()))
    
if __name__=='__main__':
    img_file1='img_compare/1080-img1.png'
    img_file2='img_compare/1080-img1_resize_x4_SRCNN_x4.png'
    img_file2='img_compare/1080-img1_resize_x4_SROP_x4.png'
    img_file2='img_compare/1080-img1_resize_x4_SROP_x4_denoised.png'
    img1=pil_image.open(img_file1).convert('RGB')
    img2=pil_image.open(img_file2).convert('RGB')
    compare(img1,img2)
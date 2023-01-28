import torch
from torch import nn as nn
from torch.nn import functional as F
import os,sys
import numpy as np
import cv2
import argparse
from time import time as ttime,sleep
import PIL.Image as pil_image
import threading,torch,os
from random import uniform
from multiprocessing import Queue
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from moviepy.editor import VideoFileClip



class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8, bias=False):
        super(SEBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction, 1, 1, 0, bias=bias)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
            x0 = torch.mean(x.float(), dim=(2, 3), keepdim=True).half()
        else:
            x0 = torch.mean(x, dim=(2, 3), keepdim=True)
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x

    def forward_mean(self, x,x0):
        x0 = self.conv1(x0)
        x0 = F.relu(x0, inplace=True)
        x0 = self.conv2(x0)
        x0 = torch.sigmoid(x0)
        x = torch.mul(x, x0)
        return x
    
class UNetConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, se):
        super(UNetConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
        )
        if se:
            self.seblock = SEBlock(out_channels, reduction=8, bias=True)
        else:
            self.seblock = None

    def forward(self, x):
        z = self.conv(x)
        if self.seblock is not None:
            z = self.seblock(z)
        return z
    
class UNet1(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet1, self).__init__()
        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 128, 64, se=True)
        self.conv2_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z

    def forward_a(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1,x2

    def forward_b(self, x1,x2):
        x2 = self.conv2_up(x2)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)

        x1 = F.pad(x1, (-4, -4, -4, -4))
        x3 = self.conv3(x1 + x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        z = self.conv_bottom(x3)
        return z
    
class UNet2(nn.Module):
    def __init__(self, in_channels, out_channels, deconv):
        super(UNet2, self).__init__()

        self.conv1 = UNetConv(in_channels, 32, 64, se=False)
        self.conv1_down = nn.Conv2d(64, 64, 2, 2, 0)
        self.conv2 = UNetConv(64, 64, 128, se=True)
        self.conv2_down = nn.Conv2d(128, 128, 2, 2, 0)
        self.conv3 = UNetConv(128, 256, 128, se=True)
        self.conv3_up = nn.ConvTranspose2d(128, 128, 2, 2, 0)
        self.conv4 = UNetConv(128, 64, 64, se=True)
        self.conv4_up = nn.ConvTranspose2d(64, 64, 2, 2, 0)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)

        if deconv:
            self.conv_bottom = nn.ConvTranspose2d(64, out_channels, 4, 2, 3)
        else:
            self.conv_bottom = nn.Conv2d(64, out_channels, 3, 1, 0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2(x2)

        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3(x3)
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)

        x2 = F.pad(x2, (-4, -4, -4, -4))
        x4 = self.conv4(x2 + x3)
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)

        x1 = F.pad(x1, (-16, -16, -16, -16))
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z

    def forward_a(self, x):#conv234结尾有se
        x1 = self.conv1(x)
        x2 = self.conv1_down(x1)
        x2 = F.leaky_relu(x2, 0.1, inplace=True)
        x2 = self.conv2.conv(x2)
        return x1,x2

    def forward_b(self, x2):  # conv234结尾有se
        x3 = self.conv2_down(x2)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)
        x3 = self.conv3.conv(x3)
        return x3

    def forward_c(self, x2,x3):  # conv234结尾有se
        x3 = self.conv3_up(x3)
        x3 = F.leaky_relu(x3, 0.1, inplace=True)

        x2 = F.pad(x2, (-4, -4, -4, -4))
        x4 = self.conv4.conv(x2 + x3)
        return x4

    def forward_d(self, x1,x4):  # conv234结尾有se
        x4 = self.conv4_up(x4)
        x4 = F.leaky_relu(x4, 0.1, inplace=True)

        x1 = F.pad(x1, (-16, -16, -16, -16))
        x5 = self.conv5(x1 + x4)
        x5 = F.leaky_relu(x5, 0.1, inplace=True)

        z = self.conv_bottom(x5)
        return z

class UpCunet2x(nn.Module):#完美tile，全程无损
    def __init__(self, in_channels=3, out_channels=3):
        super(UpCunet2x, self).__init__()
        self.unet1 = UNet1(in_channels, out_channels, deconv=True)
        self.unet2 = UNet2(in_channels, out_channels, deconv=False)

    def forward(self, x,tile_mode):#1.7G
        n, c, h0, w0 = x.shape
        if(tile_mode==0):#不tile
            ph = ((h0 - 1) // 2 + 1) * 2
            pw = ((w0 - 1) // 2 + 1) * 2
            x = F.pad(x, (18, 18 + pw - w0, 18, 18 + ph - h0), 'reflect')  # 需要保证被2整除
            x = self.unet1.forward(x)
            x0 = self.unet2.forward(x)
            x1 = F.pad(x, (-20, -20, -20, -20))
            x = torch.add(x0, x1)
            if (w0 != pw or h0 != ph): x = x[:, :, :h0 * 2, :w0 * 2]
            return x
        elif(tile_mode==1):# 对长边减半
            if(w0>=h0):
                crop_size_w=((w0-1)//4*4+4)//2#减半后能被2整除，所以要先被4整除
                crop_size_h=(h0-1)//2*2+2#能被2整除
            else:
                crop_size_h=((h0-1)//4*4+4)//2#减半后能被2整除，所以要先被4整除
                crop_size_w=(w0-1)//2*2+2#能被2整除
            crop_size=(crop_size_h,crop_size_w)#6.6G
        elif(tile_mode==2):#hw都减半
            crop_size=(((h0-1)//4*4+4)//2,((w0-1)//4*4+4)//2)#5.6G
        elif(tile_mode==3):#hw都三分之一
            crop_size=(((h0-1)//6*6+6)//3,((w0-1)//6*6+6)//3)#4.2G
        elif(tile_mode==4):#hw都四分之一
            crop_size=(((h0-1)//8*8+8)//4,((w0-1)//8*8+8)//4)#3.7G
        ph = ((h0 - 1) // crop_size[0] + 1) * crop_size[0]
        pw = ((w0 - 1) // crop_size[1] + 1) * crop_size[1]
        x=F.pad(x,(18,18+pw-w0,18,18+ph-h0),'reflect')
        n,c,h,w=x.shape
        se_mean0=torch.zeros((n,64,1,1)).to(x.device)
        if ("Half" in x.type()):
            se_mean0=se_mean0.half()
        n_patch=0
        tmp_dict={}
        opt_res_dict={}
        for i in range(0,h-36,crop_size[0]):
            tmp_dict[i]={}
            for j in range(0,w-36,crop_size[1]):
                x_crop=x[:,:,i:i+crop_size[0]+36,j:j+crop_size[1]+36]
                n,c1,h1,w1=x_crop.shape
                tmp0,x_crop = self.unet1.forward_a(x_crop)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(x_crop.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(x_crop, dim=(2, 3),keepdim=True)
                se_mean0+=tmp_se_mean
                n_patch+=1
                tmp_dict[i][j]=(tmp0,x_crop)
        se_mean0/=n_patch
        se_mean1 = torch.zeros((n, 128, 1, 1)).to(x.device)#64#128#128#64
        if ("Half" in x.type()):
            se_mean1=se_mean1.half()
        for i in range(0,h-36,crop_size[0]):
            for j in range(0,w-36,crop_size[1]):
                tmp0, x_crop=tmp_dict[i][j]
                x_crop=self.unet1.conv2.seblock.forward_mean(x_crop,se_mean0)
                opt_unet1=self.unet1.forward_b(tmp0,x_crop)
                tmp_x1,tmp_x2 = self.unet2.forward_a(opt_unet1)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x2.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x2, dim=(2, 3),keepdim=True)
                se_mean1+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x2)
        se_mean1/=n_patch
        se_mean0 = torch.zeros((n, 128, 1, 1)).to(x.device)  # 64#128#128#64
        if ("Half" in x.type()):
            se_mean0=se_mean0.half()
        for i in range(0,h-36,crop_size[0]):
            for j in range(0,w-36,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x2=tmp_dict[i][j]
                tmp_x2=self.unet2.conv2.seblock.forward_mean(tmp_x2,se_mean1)
                tmp_x3=self.unet2.forward_b(tmp_x2)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x3.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x3, dim=(2, 3),keepdim=True)
                se_mean0+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x2,tmp_x3)
        se_mean0/=n_patch
        se_mean1 = torch.zeros((n, 64, 1, 1)).to(x.device)  # 64#128#128#64
        if ("Half" in x.type()):
            se_mean1=se_mean1.half()
        for i in range(0,h-36,crop_size[0]):
            for j in range(0,w-36,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x2,tmp_x3=tmp_dict[i][j]
                tmp_x3=self.unet2.conv3.seblock.forward_mean(tmp_x3,se_mean0)
                tmp_x4=self.unet2.forward_c(tmp_x2,tmp_x3)
                if ("Half" in x.type()):  # torch.HalfTensor/torch.cuda.HalfTensor
                    tmp_se_mean = torch.mean(tmp_x4.float(), dim=(2, 3),keepdim=True).half()
                else:
                    tmp_se_mean = torch.mean(tmp_x4, dim=(2, 3),keepdim=True)
                se_mean1+=tmp_se_mean
                tmp_dict[i][j]=(opt_unet1,tmp_x1,tmp_x4)
        se_mean1/=n_patch
        for i in range(0,h-36,crop_size[0]):
            opt_res_dict[i]={}
            for j in range(0,w-36,crop_size[1]):
                opt_unet1,tmp_x1, tmp_x4=tmp_dict[i][j]
                tmp_x4=self.unet2.conv4.seblock.forward_mean(tmp_x4,se_mean1)
                x0=self.unet2.forward_d(tmp_x1,tmp_x4)
                x1 = F.pad(opt_unet1,(-20,-20,-20,-20))
                x_crop = torch.add(x0, x1)#x0是unet2的最终输出
                opt_res_dict[i][j] = x_crop
        del tmp_dict
        torch.cuda.empty_cache()
        res = torch.zeros((n, c, h * 2 - 72, w * 2 - 72)).to(x.device)
        if ("Half" in x.type()):
            res=res.half()
        for i in range(0,h-36,crop_size[0]):
            for j in range(0,w-36,crop_size[1]):
                res[:, :, i * 2:i * 2 + h1 * 2 - 72, j * 2:j * 2 + w1 * 2 - 72]=opt_res_dict[i][j]
        del opt_res_dict
        torch.cuda.empty_cache()
        if(w0!=pw or h0!=ph):res=res[:,:,:h0*2,:w0*2]
        return res#

class RealWaifuUpScaler(object):
    def __init__(self,scale,weight_path,half,device):
        weight = torch.load(weight_path, map_location="cpu")
        self.model=eval("UpCunet%sx"%scale)()
        if(half==True):self.model=self.model.half().to(device)
        else:self.model=self.model.to(device)
        self.model.load_state_dict(weight, strict=True)
        self.model.eval()
        self.half=half
        self.device=device

    def np2tensor(self,np_frame):
        if (self.half == False):return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).to(self.device).float() / 255
        else:return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).to(self.device).half() / 255

    def tensor2np(self,tensor):
        if (self.half == False):return (np.transpose((tensor.data.squeeze()* 255.0).round().clamp_(0, 255).byte().cpu().numpy(), (1, 2, 0)))
        else:return (np.transpose((tensor.data.squeeze().float()*255.0).round().clamp_(0, 255).byte().cpu().numpy(), (1, 2, 0)))

    def __call__(self, frame,tile_mode):
        with torch.no_grad():
            tensor = self.np2tensor(frame)
            result = self.tensor2np(self.model(tensor,tile_mode))
        return result
    
class UpScalerMT(threading.Thread):
    def __init__(self, inp_q, res_q, device, model,p_sleep,nt,tile):
        threading.Thread.__init__(self)
        self.device = device
        self.inp_q = inp_q
        self.res_q = res_q
        self.model = model
        self.nt = nt
        self.p_sleep=p_sleep
        self.tile=tile

    def inference(self, tmp):
        idx, np_frame = tmp
        with torch.no_grad():
            res = self.model(np_frame,self.tile)
        if(self.nt>1):
            sleep(uniform(self.p_sleep[0],self.p_sleep[1]))
        return (idx, res)

    def run(self):
        while (1):
            tmp = self.inp_q.get()
            if (tmp == None):
                # print("exit")
                break
            self.res_q.put(self.inference(tmp))
class VideoRealWaifuUpScaler(object):
    def __init__(self,nt,n_gpu,scale,half,tile,p_sleep,decode_sleep,encode_params):
        self.nt = nt
        self.n_gpu = n_gpu  # 每块GPU开nt个进程
        self.scale = scale
        self.encode_params = encode_params
        self.decode_sleep=decode_sleep

        device_base = "cuda"
        self.inp_q = Queue(self.nt * self.n_gpu * 2)  # 抽帧缓存上限帧数
        self.res_q = Queue(self.nt * self.n_gpu * 2)  # 超分帧结果缓存上限
        for i in range(self.n_gpu):
            device = device_base + ":%s" % i
            #load+device初始化好当前卡的模型
            model=RealWaifuUpScaler(self.scale, "up2x-latest-denoise1x.pth", half, device)
            for _ in range(self.nt):
                upscaler = UpScalerMT(self.inp_q, self.res_q, device, model,p_sleep,self.nt,tile)
                upscaler.start()

    def __call__(self, inp_path,opt_path):
        objVideoreader = VideoFileClip(filename=inp_path)
        w,h=objVideoreader.reader.size
        fps=objVideoreader.reader.fps
        total_frame=objVideoreader.reader.nframes
        if(objVideoreader.audio):
            tmp_audio_path="%s.m4a"%inp_path
            objVideoreader.audio.write_audiofile(tmp_audio_path,codec="aac")
            writer = FFMPEG_VideoWriter(opt_path, (w * self.scale, h * self.scale), fps, ffmpeg_params=self.encode_params,audiofile=tmp_audio_path)  # slower#medium
        else:
            writer = FFMPEG_VideoWriter(opt_path, (w * self.scale, h * self.scale), fps, ffmpeg_params=self.encode_params)  # slower#medium
        now_idx = 0
        idx2res = {}
        t0 = ttime()
        for idx, frame in enumerate(objVideoreader.iter_frames()):
            # print(1,idx, self.inp_q.qsize(), self.res_q.qsize(), now_idx, sorted(idx2res.keys()))  ##
            if(idx%10==0):
                print("total frame:%s\tdecoded frames:%s"%(int(total_frame),idx))  ##
            self.inp_q.put((idx, frame))
            sleep(self.decode_sleep)#否则解帧会一直抢主进程的CPU到100%，不给其他线程CPU空间进行图像预处理和后处理
            while (1):  # 取出处理好的所有结果
                if (self.res_q.empty()): break
                iidx, res = self.res_q.get()
                idx2res[iidx] = res
            # if (idx % 100 == 0):
            while (1):  # 按照idx排序写帧
                if (now_idx not in idx2res): break
                writer.write_frame(idx2res[now_idx])
                del idx2res[now_idx]
                now_idx += 1
        idx+=1
        while (1):
            # print(2,idx, self.inp_q.qsize(), self.res_q.qsize(), now_idx, sorted(idx2res.keys()))  ##
            # if (now_idx >= idx + 1): break  # 全部帧都写入了，跳出
            while (1):  # 取出处理好的所有结果
                if (self.res_q.empty()): break
                iidx, res = self.res_q.get()
                idx2res[iidx] = res
            while (1):  # 按照idx排序写帧
                if (now_idx not in idx2res): break
                writer.write_frame(idx2res[now_idx])
                del idx2res[now_idx]
                now_idx += 1
            if(self.inp_q.qsize()==0 and self.res_q.qsize()==0 and idx==now_idx):break
            sleep(0.02)
        # print(3,idx, self.inp_q.qsize(), self.res_q.qsize(), now_idx, sorted(idx2res.keys()))  ##
        for _ in range(self.nt * self.n_gpu):  # 全部结果拿到后，关掉模型线程
            self.inp_q.put(None)
        writer.close()
        os.remove(tmp_audio_path)
        t1 = ttime()
        print(inp_path,"done,time cost:",t1 - t0)

def img_Denoise(img):
    device="cuda:0"
    half=True
    tile=3
    img=img.resize((img.width // 2, img.height // 2), resample=pil_image.Resampling.BICUBIC)
    img=np.asarray(img)    
    upscaler2x = RealWaifuUpScaler(2, "up2x-latest-denoise1x.pth", half, device=device)
    img = upscaler2x(img, tile_mode=tile)  
    img=pil_image.fromarray(img).convert('RGB')
    return img

def video_Denoise(video_file,opt_path):
    device="cuda:0"
    half=True
    nt=1
    n_gpu=1
    p_sleep=(0.005,0.012)
    decode_sleep=0.002
    encode_params=['-crf', '18', '-preset', 'faster']
    tile=0
    video_upscaler=VideoRealWaifuUpScaler(nt,n_gpu,2,half,tile,p_sleep,decode_sleep,encode_params)
    video_upscaler(video_file,opt_path)
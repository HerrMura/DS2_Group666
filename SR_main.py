import argparse
from tqdm import tqdm
import shutil
import cv2
import numpy as np
import math
import torch
from SRCNN_method.SRCNN_app import get_srcnn
from SROP_method.SROP_app import get_srop
from Video_method.pic2video import frame2video
from Video_method.video2pic import video2frame
from Video_method.VSROP_app import srop_vid
import PIL.Image as pil_image
from PIL import ImageFilter
import moviepy.editor as mp
import skimage.io as io
from skimage import data_dir
import os
from moviepy import *
from moviepy.editor import *
from utils import tensor2img
from CUGAN import img_Denoise,video_Denoise
import time


def get_SR_result(image_file, scale, model, denoise=False):
    """ 
        Get the result of Super-Resolution based on one of the models.
        
        :param model: can be one of
            :py:data:`Bicubic`: A traditional quick interpolated Computer Vision method to resize the image.
            
            :py:data:`SRCNN`: A light and classic Super-Resolution model based on Convolutional Neural Network(CNN).
            
            :py:data:`SRPO`: A light model based on CNN, good at anti-aliasing, has better effect on images rendered by game engine. 
    """
    tic = time.perf_counter()
    image = pil_image.open(image_file).convert('RGB')
    
    if model=='SRCNN':
        SR_result=get_srcnn(image_file, scale)
        SR_result.save(image_file.name.replace('.', '_SRCNN_x{}.'.format(scale)))
        
    elif model=='SRPO':
        SR_result=get_srop(image_file, scale)
        if denoise==True:
            SR_result=img_Denoise(SR_result)
            SR_result.save(image_file.name.replace('.', '_SRPO_x{}.'.format(scale)))
        else:
            SR_result.save(image_file.name.replace('.', '_SRPO_x{}.'.format(scale)))
        
    elif model=='Bicubic':
        SR_result = image.resize((image.width * scale, image.height * scale), resample=pil_image.Resampling.BICUBIC)
        SR_result.save(image_file.name.replace('.', '_bicubic_x{}.'.format(scale)))
        
    
    toc = time.perf_counter()
    print('Consuming time:'+str(toc-tic))
    return SR_result
        
def get_Video_result(video_path, scale, denoise=False):
    tic = time.perf_counter()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size=5
        
    # extract frames from video, every frame
    frames_save_path = 'Video_method/V2F'
    print("extracting frames...")
    video2frame(video_path,frames_save_path)
    print("extracting done")
    
    # prepare frames for SR
    picture_list=frames_save_path+'/*.jpg'
    coll = io.ImageCollection(picture_list)
    num=len(coll)
    left=num
    frame_num=1
    batch_count=math.ceil(num/batch_size)
    
    lr_width=coll[0].shape[1]
    lr_height=coll[0].shape[0]
    hr_width=lr_width*scale
    hr_height=lr_height*scale
    
    # SR, batch by batch
    print("performing Super Resolution... ")
    with tqdm(total=num) as bar:
        for i in range(batch_count):
            if left>batch_size:
                frame_batch=np.zeros((batch_size,lr_height,lr_width,coll[0].shape[2])).astype(np.float32)
                sr_flat=np.zeros((batch_size,hr_height,hr_width,coll[0].shape[2])).astype(np.float32)
                left=left-batch_size
            else:
                frame_batch=np.zeros((left,lr_height,lr_width,coll[0].shape[2])).astype(np.float32)
                sr_flat=np.zeros((left,hr_height,hr_width,coll[0].shape[2])).astype(np.float32)
                left=0
            for j in range(frame_batch.shape[0]):
                image = pil_image.fromarray(coll[i*batch_size+j]).convert('RGB')
                #lr=image.filter(ImageFilter.SHARPEN)
                lr=image
                lr = np.array(lr).astype(np.float32)
                frame_batch[j]=lr
                flat = image.resize((hr_width, hr_height), resample=pil_image.LANCZOS)
                flat = np.array(flat).astype(np.float32)
                sr_flat[j]=flat
            
            frame_batch=torch.from_numpy(frame_batch).permute(0,3,1,2).to(device)
            sr_flat=torch.from_numpy(sr_flat).permute(0,3,1,2).to(device)
            SR_result=srop_vid(frame_batch,sr_flat,scale)
            if denoise==True:
                for k in range(SR_result.shape[0]):
                    sr_img=SR_result[k]
                    sr_img=tensor2img(sr_img)
                    sr_img=sr_img.resize((sr_img.width // 2, sr_img.height // 2), resample=pil_image.Resampling.BICUBIC)  
                    sr_img.save('Video_method/VSRPO/frame'+str(frame_num)+'.jpg')
                    frame_num=frame_num+1
                    bar.update(1)
            else:
                for k in range(SR_result.shape[0]):
                    sr_img=SR_result[k]
                    sr_img=tensor2img(sr_img)
                    sr_img.save('Video_method/VSRPO/frame'+str(frame_num)+'.jpg')
                    frame_num=frame_num+1
                    bar.update(1)
    print("SR done")
    
    print("merging frames into video...")
    # frame2video
    im_dir = 'Video_method/VSRPO/'
    video_output_dir = 'Output/output.mp4' # path of the output video
    fps = 30
    frame2video(im_dir,video_output_dir,fps)
    
    print("merging audio...")
    # extract audio
    # load the video
    video_input = VideoFileClip(video_path)
    # load the audio and save it
    video_input.audio.write_audiofile("Output/output.mp3")
    # load .mp3 file
    audio = AudioFileClip("Output/output.mp3")
    # merge the video and the audio
    video=VideoFileClip(video_output_dir)
    new_clip = video.set_audio(audio)
    # save as new .mp4
    new_clip.write_videofile("Output/Final.mp4")
    
    if denoise==True:
        print("Denoising...")
        video_Denoise("Output/Final.mp4","Output/Final_denoised.mp4")
        os.remove('Output/Final.mp4')
        print("Denoising done")
    
    os.remove('Output/output.mp4')
    os.remove('Output/output.mp3')
    if not os.path.exists('Video_method/V2F'):
        os.mkdir('Video_method/V2F')
    else:
        shutil.rmtree('Video_method/V2F')
        os.mkdir('Video_method/V2F')

    if not os.path.exists('Video_method/VSRPO'):
        os.mkdir('Video_method/VSRPO')
    else:
        shutil.rmtree('Video_method/VSRPO')
        os.mkdir('Video_method/VSRPO')
    
    toc = time.perf_counter()    
    print('Consuming time:'+str(toc-tic))
    # the final SR video is saved as 'Output/Final.mp4'    

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    parser.add_argument('--scale', type=int, required=True)
    parser.add_argument('--model', type=str, default='SRPO')
    parser.add_argument('--type', type=str, default='image')
    args = parser.parse_args()
    denoise=False
    if args.type=='image':        
        SR_result=get_SR_result(args.file, args.scale, args.model, denoise)
    elif args.type=='video':
        Vid_result=get_Video_result(args.file, args.scale, denoise)
    else:
        print("Please assign the type as image or video")
    
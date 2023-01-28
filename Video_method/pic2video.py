import cv2
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
 
 
def frame2video(im_dir,video_dir,fps):
 
    im_list = os.listdir(im_dir)
    im_list.sort(key=lambda x: int(x.replace("frame","").split('.')[0]))  # please make sure the images are in the right order
    img = Image.open(os.path.join(im_dir,im_list[0]))
    img_size = img.size # resolution of the video，images under path im_dir should be of the same size
 
 
    # fourcc = cv2.cv.CV_FOURCC('M','J','P','G') # opencv version 2
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # opencv version 3
    videoWriter = cv2.VideoWriter(video_dir, fourcc, fps, img_size)
    # count = 1
    with tqdm(total=len(im_list)) as bar:
        for i in im_list:
            im_name = os.path.join(im_dir+i)
            frame = cv2.imdecode(np.fromfile(im_name, dtype=np.uint8), -1)
            videoWriter.write(frame)
            bar.update(1)
            # count+=1
            # if (count == 200):
            #     print(im_name)
            #     break
    videoWriter.release()
    print('finish')
 
if __name__ == '__main__':
    im_dir = 'video_srop/' # save path of the frames
    video_dir = 'video_output/output.mp4' # save path of the video
    fps = 30 # frames per second
    frame2video(im_dir, video_dir, fps)

import cv2
from tqdm import tqdm

def video2frame(videos_path,frames_save_path):
 
  '''
  :param videos_path: path of the video
  :param frames_save_path: save path of the frames
  :return:
  '''
  vidcap = cv2.VideoCapture(videos_path)
  frames_num = vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
  print("total frames: "+str(frames_num))
  success, image = vidcap.read()
  count = 0
  with tqdm(total=frames_num) as bar:
    while success:
      count += 1
      cv2.imencode('.jpg', image)[1].tofile(frames_save_path + "/frame%d.jpg" % count)
      success, image = vidcap.read()
      bar.update(1)
  print("frames extracted: "+str(count))
 
if __name__ == '__main__':
   videos_path = 'img/v1.MOV'
   frames_save_path = 'video_save'
   video2frame(videos_path, frames_save_path)

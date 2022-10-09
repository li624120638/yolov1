import cv2
import os
import numpy as np
import subprocess

def draw_text(img, text, position=(0, 0), size=1, color=(255,255,255), thickness=2):
    position = (position[0], position[1]+size*30)

    cv2.putText(img, text,
        position, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        size,
        color[::-1],
        thickness)

def write_video(video_name, images):
    height, width, _ = images[0].shape
    video = cv2.VideoWriter(video_name, 0, 1, (width,height))

    for image in images:
        video.write(image)

    video.release()

def images_to_video(images_path='img%05d.bmp', video_path='video.mp4', fps=30):
    # os.system("ffmpeg -r {} -i img%01d.png -vcodec mpeg4 -y movie.mp4")
    subprocess.call("ffmpeg -r {} -i {} -vcodec mpeg4 -y {}".format(fps, images_path, video_path))

def cv2_imread(file_path):
    image_imdecode = cv2.imdecode(np.fromfile(
        file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return image_imdecode

def cv2_imwrite(file_path, img):
    _, ext = os.path.splitext(file_path)
    cv2.imencode(ext, img)[1].tofile(file_path)

def draw_bounding_box(image, color=(255,0,0), padding=5, thickness=2):
    height, width, _ = image.shape
    cv2.rectangle(image, (padding,padding), (width-padding, height-padding), color, thickness)

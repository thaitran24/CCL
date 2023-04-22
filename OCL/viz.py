import os
import cv2
import numpy as np
import glob
import imutils
from PIL import Image
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic

def load_image(image_path, model=None):
    pattern = 'gbrg'
    img = Image.open(image_path)
    img = demosaic(img, pattern)
    return np.array(img).astype(np.uint8)

def play_vis(image_path, video_name, fps=15, size=256):
    lines = glob.glob('{}/*'.format(image_path))
    lines = sorted(lines)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('{}.mp4'.format(video_name), fourcc, fps, (size, size))

    print('Processing {} frames'.format(len(lines)))

    for filename in lines:
        if not os.path.exists(filename):
            print("in image: {} not exists".format(filename))        
            continue
        
        image = imutils.resize(cv2.imread(filename), size)

        stack = np.hstack([image])
        out.write(stack)
        
    out.release()
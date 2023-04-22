import os
import cv2
import glob
import argparse
import numpy as np
from PIL import Image
from ocl import run_ocl_np, run_ocl_pil, vis_run_ocl
from viz import play_vis

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--image_path', type=str, default='images/', help='path to image(s) need to process')
arg_parser.add_argument('--step', type=int, default=1, help='save visualization step to path')
arg_parser.add_argument('--save_dir', type=str, default='results/', help='path to save processed image(s)')
arg_parser.add_argument('--save_vis', type=str, help='save visualization step to path')
arg_parser.add_argument('--run_pil', action='store_true', help='use PIL Image instead of numpy array')

args = arg_parser.parse_args()

if __name__=='__main__':
    img_paths = glob.glob('{}*'.format(args.image_path))
    total_time = []
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    for step in range(args.step):
        for pth in img_paths:
            image = cv2.imread(pth)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if args.save_vis:
                if not os.path.exists(args.save_vis):
                    os.makedirs(args.save_vis)
                labels, out_image, elapsed_time = vis_run_ocl(image, args.save_vis)
                play_vis(args.save_vis, '{}/vis_{}'.format(args.save_dir, pth.split('/')[-1].split('.')[0]))
            elif not args.run_pil:
                labels, out_image, elapsed_time = run_ocl_np(image)
            else:
                labels, out_image, elapsed_time = run_ocl_pil(Image.fromarray(image).convert('1'))
                
            total_time.append(elapsed_time)
            print('Processed: {} - Elapsed: {} s'.format(pth.split('/')[-1], elapsed_time))
            if not args.run_pil:
                cv2.imwrite(os.path.join(args.save_dir, pth.split('/')[-1]), out_image)
            else:
                cv2.imwrite(os.path.join(args.save_dir, pth.split('/')[-1]), np.asarray(out_image))

    print('\nAverage time: {} s'.format(sum(total_time) / len(total_time)))
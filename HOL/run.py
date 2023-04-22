import os
import cv2
import glob
import argparse

from hol import run_hol

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('--image_path', type=str, default='images/', help='path to image(s) need to process')
arg_parser.add_argument('--step', type=int, default=1, help='save visualization step to path')
arg_parser.add_argument('--save_dir', type=str, default='results/', help='path to save processed image(s)')

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
            out_image, elapsed_time = run_hol(image)
            total_time.append(elapsed_time)
            print('Processed: {} - Elapsed: {} s'.format(pth.split('/')[-1], elapsed_time))
            cv2.imwrite(os.path.join(args.save_dir, pth.split('/')[-1]), out_image)

    print('\nAverage time: {} s'.format(sum(total_time) / len(total_time)))
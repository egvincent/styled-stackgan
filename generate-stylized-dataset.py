# RUN IN 3.6 (conda: stylegan) from main directory (will automatically `cd` to neural-style)

import argparse
import subprocess
import os
from PIL import Image

# example:
# python generate-stylized-dataset.py \
#   --inputs ############# todo: switch to birds dataset
#   --outputs ~/project/data/coco-stylized/1-starry-night \
#   --style ~/project/styles/1-starry-night.png

parser = argparse.ArgumentParser()
parser.add_argument('--inputs', type=str, help='ABSOLUTE path to input image directory')
parser.add_argument('--outputs', type=str, help='ABSOLUTE path to output image directory')
parser.add_argument('--style', type=str, help='ABSOLUTE path to style image')
# 310 px is a magic number: it's a little over the minimum size inputs to StackGAN (Stage II) can be,
#   given that StackGAN will resize input images to 76/64 * 256 = 304 for Stage II to make
#   random cropping (to size 256) work out.
parser.add_argument('--out_size', type=int, const=310, help='optional: output image size (pixels)')
# default: stop after 500 images for the given style
parser.add_argument('--count', type=int, const=500, help='number of images to stylize')
args = parser.parse_args()

# change directory to neural-style while running
subprocess.run("cd ~/project/neural-style/", shell=True)

# iterate over images in args.inputs until hitting max of args.counts
for i,filename in os.listdir(args.inputs):
    if i >= args.count:
        print("========== done ==========")
        break

    print("========== starting neural-style for image " + str(i) + ": " + image + "==========")

    # make temporary resized image file
    img = Image.open(args.inputs + '/' + filename)
    img = img.resize((args.out_size, args.out_size), PIL.Image.BILINEAR)
    temp_file_path = "TEMP-" + filename
    img.save(temp_file_path)

    subprocess.run("python neural_style.py --styles " + args.style \
        + " --content " + temp_file_path \
        + " --output " + args.outputs + '/' + image \
        + " -- preserve-colors", \
        shell=True)

    # delete temporary resized image file
    subprocess.run("rm " + temp_file_path, shell=True)

# switch back to project directory
subprocess.run("cd ~/project/", shell=True)

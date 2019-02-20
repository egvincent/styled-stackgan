# runs StackGAN and style-transfer sequentially in their out-of-the-box default configuation
# this file is called from run_baseline.sh, once in python 2.7, once in python 3.6 ... sorry

from __future__ import print_function
import argparse
import subprocess
import os

# store a hardcoded list of images to run through style transfer, because it is very slow
content_images = ["70.png", "99.png", "154.png", "174.png", "207.png", "289.png", "904.png", "1033.png", "1301.png"]

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='either "generate" or "style"')
parser.add_argument('--style', type=str, help='the style image file')
args = parser.parse_args()

if args.mode == "generate":
	# note: in this case, we are in python 2.7
	print("========== starting StackGAN ==========")
	subprocess.call("(cd ~/project/StackGAN-Pytorch/code/ && python main.py --cfg cfg/coco_eval.yml --gpu 0 && cd ~/project/)", shell=True)

if args.mode == "style":
	# note: in this case, we are in python 3.6
	print("========== starting neural-style ==========")
	for image in content_images:
		subprocess.run("(cd ~/project/neural-style/ && python neural_style.py --styles " + args.style + " --content ~/project/StackGAN-Pytorch/models/coco/netG_epoch_90/" + image + " --output ~/project/baseline-output/styled-" + image + " && cd ~/project/)", shell=True)

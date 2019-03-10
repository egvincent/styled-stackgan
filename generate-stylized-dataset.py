# RUN IN 3.6 (conda: stylegan) from main directory (will automatically `cd` to neural-style)

import argparse
import subprocess
import os
import pickle
import scipy.misc
from PIL import Image

# example:
# python generate-stylized-dataset.py \
#   --inputs ~/project/StackGAN/Data/birds/train/304images.pickle \
#   --outputs ~/project/data/birds-stylized/1-starry-night \
#   --style ~/project/style-images/1-starry-night.jpg


# output pickle file will be placed into the same directory as the input pickle file, but
#   with suffix "-stylized"

# if you interrupt the script, delete the temporary file from neural-style


# if you already have a folder of stylized images, and just want to add more, that is fine:
#   just list the same folder as the output folder, and the script adds more to it. it will
#   re-pickle the images in the output folder into a new pickle file.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=str, help='ABSOLUTE path to input image pickle file')
    parser.add_argument('--outputs', type=str, help='ABSOLUTE path to output image directory')
    parser.add_argument('--style', type=str, help='ABSOLUTE path to style image')
    # default: start at the first image (index 0) in the input list.
    #   NOTE: the images need to be consecutive, starting from the beginning, or else the indices
    #   won't line up with the metadata indices used by StackGAN
    parser.add_argument('--start', type=int, default=0, help='index of first image to stylize')
    # default: stop after 500 images for the given style
    parser.add_argument('--count', type=int, default=500, help='number of images to stylize')
    args = parser.parse_args()
    
    # change directory to neural-style while running
    with cd("~/project/neural-style/"):
         
        # read images
        print("reading input image pickle file")
        images = None
        with open(args.inputs, 'rb') as f:
            # specify encoding to handle objects pickled in python 2.7
            images = pickle.load(f, encoding='latin1')
        print("successfully loaded images")
        
        
        for i in range(args.start, min(args.start + args.count, len(images))):        
            print("========== starting neural-style for image " + str(i) + "==========")
            
            # make temporary image file from the current image array
            temp_img_filename = "TEMP-image-" + str(i) + ".png"
            scipy.misc.imsave(temp_img_filename, images[i])
            
            # run style transfer on temporary image
            out_img_filename = "stylized-image-" + str(i) + ".png"
            subprocess.run("python neural_style.py --styles " + args.style \
                + " --content " + temp_img_filename \
                + " --output " + args.outputs + '/' + out_img_filename \
                + " --preserve-colors", \
                shell=True)
            
            # delete temporary resized image file
            subprocess.run("rm " + temp_img_filename, shell=True)

    print("========== done styling input images ==========")

    print("creating pickle file")
    image_list = []
    for filename in os.listdir(args.outputs):
        image = scipy.misc.imread(args.outputs + '/' + filename)
        image_list.append(image)
    input_filename = args.inputs.split(".")[0]
    output_filename = input_filename + "-stylized.pickle"
    with open(output_filename, 'wb') as f:
        pickle.dump(image_list, f, protocol=2)  # protocol=2 is so it will work when it's read in python 2.7

    print("done: created " + output_filename)


# need this to change directory safely
# from https://stackoverflow.com/a/13197763
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)
        
    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


if __name__ == "__main__":
    main()

# RUN IN 3.6 (conda: stylegan) from main directory (will automatically `cd` to neural-style)

import argparse
import subprocess
import os
import pickle
import scipy.misc
import numpy as np
from PIL import Image

# Generates stylized input images and truncated pickle files for each type of data, to
#   match the number or stylized images. Does NOT randomize, but instead is intended to 
#   called in such a way that the outputs are K full species' worth of stylized images
#   (such that we have enough training data per species, instead of a scattered set of
#   images across many species).

# example:
# python generate-stylized-dataset.py \
#   --input ~/project/StyleGAN/Data/birds/train/ \
#   --image_out ~/project/data/birds-stylized-images/ \
#   --pickle_out ~/project/data/birds-stylized/ \
#   --style ~/project/style-images/ \
#   --test_percent 30 \
#   --stop_index 450

# input and pickle_out directory should/will have the following pickle files, divided into train and test folders:
#   - 76images.pickle, a list of np arrays of shape (76, 76, 3)
#   - 304images.pickle, a list of np arrays of shape (304, 304, 3)
#   - char-CNN-RNN-embeddings.pickle, a list of np arrays of shape (10, 1024)
#   - class_info.pickle, a list of ints (class IDs)
#   - filenames.pickle, a list of strings
# pickle_out directory will also have:
#   - style_info.pickle, a list of ints (style IDs)

# if you interrupt the script, delete the temporary file from neural-style/


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input', type=str, help='ABSOLUTE path to input data directory')
    parser.add_argument('--image_out', type=str, help='ABSOLUTE path to output image directory')
    parser.add_argument('--pickle_out', type=str, help='ABSOLUTE path to output pickle file directory')
    parser.add_argument('--styles', type=str, help='ABSOLUTE path to style image directory')
    parser.add_argument('--num_new', type=int, help='number of new images to stylize and pickle \
            per style. 0 to just repickle. don\'t pass --stop_index at the same time')
    parser.add_argument('--stop_index', type=int, help='alternative to --num_new: index of last image to stylize, \
            for each style, instead of a number to add. no bounds checking, so don\'t be dumb. \
            use this option if you currently have a different number of images of each style.')
    parser.add_argument('--test_percent', type=int, help='percent of the data to put in the \
            test set (only the pickle output is divided into train/test)')

    args = parser.parse_args()


    # generate args.count new stylized images and put them in args.image_out
    # -------------------------------------------------------------------------
    
    print("reading input image pickle file")
    images = None
    with open(os.path.join(args.input, "304images.pickle"), 'rb') as f:
        # specify encoding to handle objects pickled in python 2.7
        images = pickle.load(f, encoding='latin1')
    print("done")

    style_names = []  # just save names for the pickle file creation step later
    num_existing = None  # same
    stop_index = None  # same

    for style_filename in os.listdir(args.styles):

        print("making output image directory, if doesn't already exist")
        style_name = style_filename[: style_filename.rfind(".")]
        style_names.append(style_name)
        subprocess.run("mkdir -p " + os.path.join(args.image_out, style_name), shell=True)
        print("done")

        print("========= starting neural-style for style " + style_name + " ==========")

        num_existing = len(os.listdir(os.path.join(args.image_out, style_name)))
        stop_index = args.stop_index
        if stop_index == None:
            stop_index = min(num_existing + args.num_new - 1, len(images) - 1)

        # change directory to neural-style while running
        with cd("~/project/neural-style/"):

            for i in range(num_existing, stop_index + 1):
                print("---------- starting neural-style for image " + str(i) + " ----------")
                
                # make temporary image file from the current image array
                temp_img_filename = "TEMP-image-" + str(i) + ".png"
                scipy.misc.imsave(temp_img_filename, images[i])
                
                # run style transfer on temporary image
                out_img_filename = "stylized-image-" + str(i).zfill(5) + ".png"
                subprocess.run("python neural_style.py " \
                    + " --styles " + os.path.join(args.styles, style_filename) \
                    + " --content " + temp_img_filename \
                    + " --output " + os.path.join(args.image_out, style_name, out_img_filename) \
                    + " --preserve-colors", \
                    shell=True)
                
                # delete temporary resized image file
                subprocess.run("rm " + temp_img_filename, shell=True)

                print("done")

        print("done")

    print("========== done styling input images ==========")


    # pickle all the data from each data file to be of the same length as the 
    #   total number of stylized images, and put the results in args.pickle_out
    # -------------------------------------------------------------------------

    print("========== creating pickle files ==========")
    
    # for each pickle.read, use encoding=latin, so it will work with pickles made in python 2.7
    # for each pickle.dump, use protocol=2, so it will work when read in python 2.7
    # also note that I repeatedly 

    total_per_style = None
    if args.num_new != None:
        # (assumes the same number of input images in each style)
        total_per_style = num_existing + args.num_new
    else:
        total_per_style = stop_index + 1

    pickle_queue = []

    hr_images = []
    for style_name in style_names:
        for filename in os.listdir(os.path.join(args.image_out, style_name)):
            image = scipy.misc.imread(os.path.join(args.image_out, style_name, filename))
            hr_images.append(image)
    pickle_queue.append(("304images.pickle", hr_images))

    style_info_nested = [[i] * total_per_style for i,_ in enumerate(style_names)]
    # flatten nested list with this dumb notation: https://stackoverflow.com/a/952952
    style_info = [item for sublist in style_info_nested for item in sublist]
    pickle_queue.append(("style_info.pickle", style_info))

    # for all the following, truncate and replicate the same content for each style, 
    #   because nothing changes besides the styled image and the style index

    lr_images_single = None
    with open(os.path.join(args.input, "76images.pickle"), 'rb') as f:
        lr_images_single = pickle.load(f, encoding='latin1')[:total_per_style]
    lr_images = lr_images_single * len(style_names)
    pickle_queue.append(("76images.pickle", lr_images))
    
    embeddings_single = None
    with open(os.path.join(args.input, "char-CNN-RNN-embeddings.pickle"), 'rb') as f:
        embeddings_single = pickle.load(f, encoding='latin1')[:total_per_style]
    embeddings = embeddings_single * len(style_names)
    pickle_queue.append(("char-CNN-RNN-embeddings.pickle", embeddings))

    class_info_single = None
    with open(os.path.join(args.input, "class_info.pickle"), 'rb') as f:
        class_info_single = pickle.load(f, encoding='latin1')[:total_per_style]
    class_info = class_info_single * len(style_names)
    pickle_queue.append(("class_info.pickle", class_info))

    filenames_single = None
    with open(os.path.join(args.input, "filenames.pickle"), 'rb') as f:
        filenames_single = pickle.load(f, encoding='latin1')[:total_per_style]
    filenames = filenames_single * len(style_names)
    pickle_queue.append(("filenames.pickle", filenames))

    
    # print(str(len(hr_images)))
    # print(str(len(style_info)))
    # print(str(len(lr_images)))
    # print(str(len(embeddings)))
    # print(str(len(class_info)))
    # print(str(len(filenames)))

    assert len(hr_images) == len(style_info) == len(lr_images) \
            == len(embeddings) == len(class_info) == len(filenames)

    
    print("making output pickle directories (for train and test), if they don't already exist")
    subprocess.run("mkdir -p " + os.path.join(args.pickle_out, "train"), shell=True)
    subprocess.run("mkdir -p " + os.path.join(args.pickle_out, "test"), shell=True)
    print("done")


    train_frac = 1 - args.test_percent / 100.0
    test_start_index = int(train_frac * len(hr_images))
    indices = np.random.permutation(len(hr_images))
    train_indices = indices[:test_start_index]
    test_indices = indices[test_start_index:]

    for name, data in pickle_queue:
        with open(os.path.join(args.pickle_out, "train", name), 'wb') as f:
            pickle.dump(np.take(data, train_indices), f, protocol=2)
        with open(os.path.join(args.pickle_out, "test", name), 'wb') as f:
            pickle.dump(np.take(data, test_indices), f, protocol=2)
        print("created " + name)



    print("========== done creating pickle files ==========")



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

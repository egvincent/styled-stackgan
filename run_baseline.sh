#!/usr/bin/env bash

# a wrapper to call StackGAN and neural-style in their respective conda environments (python 2.7 vs 3.6, for instance)

# run StackGAN on all images in the COCO evaluation set
source activate stackgan
python ~/project/baseline.py --mode generate
source deactivate

# run neural-style on selected images
source activate stylegan
python ~/project/baseline.py --mode style --style examples/1-style.jpg
source deactivate

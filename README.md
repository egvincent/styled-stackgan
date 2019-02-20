### Baseline

Run with `./run_baseline.sh`
* this calls the python script `baseline.py`. see file for more information
* make sure the directory `baseline-output/` exists before running
  * outputs will be placed here. see `StackGAN-Pytorch/data/coco/test/val_captions.txt` for captions (by line number)

Baseline relies on submodules:
* https://github.com/hanzhanggit/StackGAN-Pytorch
  * runs in python 2.7 and uses pytorch
  * we have it configured to work in the conda environment `stackgan`
* https://github.com/anishathalye/neural-style
  * runs in python 3.6 and uses tensorflow
  * we have it configured to work in the conda environment `stylegan`
    * (old name, but conda environments can't be renamed)

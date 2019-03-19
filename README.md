### Conda environments

To handle miscellaneous conflicting dependencies, we use conda environments, provided with this repository.  
* to activate:
    * install with `conda env create -f <environment file name>.yml`
    * activate with `source activate <environment name>`
* to make changes (needs to be activated)
    * install new packages:	`conda install <package>`
    * export to file:		`conda env export > <environment file name>.yml`

environments included:
* `style_env.yml` is for neural-style, with environment `stylegan` (old name, sorry)
    * this is python 3.6
* `stackgan_env.yml` is for StackGAN-Pytorch and StyleGAN-Pytorch, with environment `stackgan`
    * this is python 2.7
    * note: contains pip packages that had no conda version: gdown, easydict, torchfile, tensorboard-pytorch, pyyaml. if any don’t work, get with pip install
* `stackgan_tf_enf.yml` is for StackGAN, StyleGAN, and StackGAN-inception-model, with environment `stackgan-tf`
    * also python 2.7
    * pip packages: gdown, prettytensor==0.7.4, progressbar, python-dateutil, easydict, pandas, torchfile,  tensorflow-gpu==1.0.1

### Required setup

The large data and pretrained model files haven't all been included with the repository (and submodules)

For StackGAN (tensorflow version):
* set up torch so we can use the pretrained char-cnn-rnn embeddings:
    * http://torch.ch/docs/getting-started.html#_ 
    * but actually before the “./install.sh” step, do this
* add the conda environment’s library directory to torch’s path environment variable, so it can find the CUDA files 
    * `export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/<path to anaconda3 directory>/anaconda3/envs/stackgan-tf/lib"`
* other notes about the repository:
    * tensorflow has been updated from 0.12 to 1.0.1 to handle versioning conflicts with other packages. other related changes have been made, as listed here: https://github.com/hanzhanggit/StackGAN/issues/13#issuecomment-300400165
    * this involved moving run_exp.py from stageI/ and stageII/ to run_exp_stageI.py and run_exp_stageII.py in the main directory
* data/embeddings/model:
    * put pretrained char-cnn-rnn embeddings for birds in Data/ (and unzip it)
        * gdown https://drive.google.com/uc?id=0B3y_msrWZaXLT1BZdVdycDY5TEE 
        * (these are text embeddings made by a text encoder, like the one below)
    * put pretrained model on the char-cnn-rnn model for birds in models/
        * gdown https://drive.google.com/uc?id=0B3y_msrWZaXLNUNKa3BaRjAyTzQ 
    * for the demo, download pretrained text encoder into models/text-encoder:
        * gdown https://drive.google.com/uc?id=0B0ywwgffWnLLU0F3UHA3NzFTNEE 
    * get bird image data (will be preprocessed and used to generate stylized versions. skip if using the stylized images provided with this repo is sufficient):
        * wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz 
* preprocessing bird data:	(this crops them based on the bounding boxes)
    * need to add the project repo to the pythonpath so there aren't import issues
        * `export PYTHONPATH="${PYTHONPATH}:/<path to repo>/StackGAN"`
    * then just run `misc/preprocess_birds.py`
    * they are put in Data/birds/train and Data/birds/test, and we can reuse this to create the styled dataset

For reed 2016 text embeddings:
* mkdir data/ in the repo, and add then get bird data:
    * gdown https://drive.google.com/uc?id=0B0ywwgffWnLLZW9uVHNjb2JmNlE
    * then extract into a folder called data/cub/
* edit scripts/train_cub_hybrid.sh to change gpuid to 0 (or the corresponding correct ID for your system. it's just hardcoded in there)
* train with by running scripts/train_cub_hybrid.sh from the main directory for this repo. checkpoints should be output to cv/

For neural-style:
* in main directory:
    * `wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat`

For StackGAN-inception-model
* in directory: download and unzip:
    * gdown https://drive.google.com/uc?id=0B3y_msrWZaXLMzNMNWhWdW0zVWs 
* to run, you need to pass the model file and the test images directory, so:
    * `python inception_score.py --image_folder ~/project/data/birds-stylized-images --checkpoint_dir <path to .ckpt file> --gpu <gpu id>`


**old:**

For StackGAN-pytorch (not using this anymore)
* in data/coco:
    * gdown https://drive.google.com/uc?id=0B3y_msrWZaXLQXVzOENCY2E3TlU
    * gdown https://drive.google.com/uc?id=0B3y_msrWZaXLeEs5MTg0RC1fa0U
    * wget http://images.cocodataset.org/zips/train2017.zip 
    * (and unzip all)
* in models/coco:
    * gdown https://drive.google.com/uc?id=0B3y_msrWZaXLYjNra2ZSSmtVQlE
        * rename to netG_epoch_90.pth
 

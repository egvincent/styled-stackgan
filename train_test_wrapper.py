# a wrapper module for our entire system for use in bayesian hyperparameter optimization.
# run in stackgan-tf environment (2.7)

import subprocess
import os
import glob
from util import cd


# black box function that takes hyperparameters and returns the inception score.
def run_model(z_dim, discriminator_lr, generator_lr, lr_decay_epoch, \
        num_embedding, coeff_KL, embedding_dim, gf_dim, df_dim): 
    # note: this takes as arguments all hyperparameters it would be reasonable
    #   to tune with enough time, but one can always set a 1-item search "range" 
    #   in the constrained optimization process to opt out of optimizing them.
    
    # Hardcoded constants (for now)
    max_epoch = 10
    snapshot_interval = int(0.3 * 451) * max_epoch - 1
    batch_size = 64  # may need to adjust
    num_copy = 8  # may need to adjust
    test_images_dir = "/home/ubuntu/project/data/birds-stylized-images/1-starry-night/"  ### TODO: make work with all styles
    gpu_id = 0
    
    cfg_path = "/home/ubuntu/project/StackGAN-inception-model/tmp-cfg.yml"
    
    
    # Step 1: train model for max_epoch epochs
    # ---------------------------------
    
    # build config file for training (a temporary file)
    with open(cfg_path, 'w') as f:
        f.write("CONFIG_NAME: 'stageII'\n")
        f.write("DATASET_NAME: 'birds'\n")
        f.write("EMBEDDING_TYPE: 'cnn-rnn'\n")
        f.write("GPU_ID: 0\n")
        f.write("Z_DIM: " + str(z_dim) + "\n")
        f.write("TRAIN: \n")
        f.write("    FLAG: True\n")
        f.write("    PRETRAINED_MODEL: './ckt_logs/birds/stageI/model_82000.ckpt'\n")
        f.write("    PRETRAINED_EPOCH: 600\n")
        f.write("    BATCH_SIZE: 64\n")
        f.write("    NUM_COPY: 4\n")
        f.write("    MAX_EPOCH: " + str(max_epoch) + "\n")
        f.write("    SNAPSHOT_INTERVAL : " + str(snapshot_interval) + "\n")
        f.write("    DISCRIMINATOR_LR: " + str(discriminator_lr) + "\n")
        f.write("    GENERATOR_LR: " + str(generator_lr) + "\n")
        f.write("    LR_DECAY_EPOCH: " + str(lr_decay_epoch) + "\n")
        f.write("    NUM_EMBEDDING: " + str(num_embedding) + "\n")
        f.write("    COEFF:\n")
        f.write("      KL: " + str(coeff_KL) + "\n")
        f.write("GAN:\n")
        f.write("    EMBEDDING_DIM: " + str(embedding_dim) + "\n")
        f.write("    GF_DIM: " + str(gf_dim) + "\n")
        f.write("    DF_DIM: " + str(df_dim) + "\n")
    
    # train with this config
    with cd("/home/ubuntu/project/StyleGAN/"):
        subprocess.call("python run_exp_stageII.py --gpu 0 --cfg " + cfg_path)
    # this should result in a new directory in ckt_logs/birds/, and that folder will have
    #   a checkpoint called model_<snapshot_interval>.ckpt (with some suffix we can ignore if 
    #   we use saver.restore, in theory)

    
    # Step 2: get and return inception score
    # --------------------------------------

    # get checkpoint filename as described above
    log_dir = "~/project/ckt_logs/birds/stageII/"
    checkpoint_list = glob.glob(log_dir + "*")
    latest_checkpoint = max(checkpoint_list, key=os.path.getctime)
    checkpoint_path = os.path.join(log_dir, latest_checkpoint, \
            "model_" + str(snapshot_interval) + ".ckpt")   # may need to adjust

    # the inception-model repo takes a test image folder and a 

    with cd("/home/ubuntu/project/StackGAN-inception-model/"):
        subprocess.call("python inception_score.py --checkpoint_dir " + checkpoint_path \
                + " --image_folder " + test_images_dir + " --gpu " + str(gpu_id))
    # NOTE: this is currently broken, because the model checkpoint format seems to be different than the format
    #   expected by the inception model.
    # TODO also: parse the values printed by this command. there's a mean inception score printed we need.
    #   probably use ">" if supported by subprocess.call (2.7 version without shell=True)

    
    # (then remove the temporary file)
    subprocess.call("rm " + cfg_path)

    # then return the inception score


# draft of bayesian hyperparameter search script
# see https://github.com/fmfn/BayesianOptimization

# not currently operational due to issues in train_test_wrapper
#   (ckpt file format compatibility) that we aren't bothering to fix
#   because we don't actually have time to run this at any meaningful length.

from bayes_opt import BayesianOptimization
import train_test_wrapper

# define bounds of constrained optmization
# if range is a single value, we aren't optimizing that hyperparameter
pbounds = {  # TODO
    # Z_DIM is the size of the gaussian noise vector
    'z_dim': (50, 150),  # example
    # DISCRIMINATOR_LR and GENERATOR_LR are learning rates for discriminator and generator
    'discriminator_lr': (0.0002, 0.0002),
    'generator_lr': (0.0002, 0.0002),
    # LR_DECAY_EPOCH is the # epochs between each time it decays the learning rate
    'lr_decay_epoch': (100, 100),
    # NUM_EMBEDDING is the number of caption embeddings per image that they FC-layer down to embedding_dim
    'num_embedding': (4, 4),
    # EMBEDDING_DIM is self explanatory
    'embedding_dim': (128, 128),
    # GF_DIM is the generator feature vector length, like the depth of the downscaled image volume (kinda. actually it's this divided by 4)
    'gf_dim': (128, 128),
    # DF_DIM is similar but for the discriminator (and it's divided by 8)
    'df_dim': (64, 64),
    # COEFF.KL is the kullbach-liebler coefficient for the conditioning augmentation
    'coeff_KL': (2.0, 2.0)
}

optimizer = BayesianOptimization(
    f=train_test_wrapper.run_model,
    pbounds=pbounds,
    random_state=1
)

# init_points: How many steps of random exploration you want to perform. 
#   Random exploration can help by diversifying the exploration space.
# n_iter: How many steps of bayesian optimization you want to perform. 
#   The more steps the more likely to find a good maximum you are.
optimizer.maximize(init_points=10, n_iter=100)

print("best values found over 100 iterations of bayesian optimization:")
print(optimizer.max)


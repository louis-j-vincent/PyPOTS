import sys
sys.path.append("..")

from PyPOTS_gp_benchmark_helpers import artifical_GP_generation

n_time_points = 30
n_new_dims = 10
dataset_train_nan, dataset_val_nan, dataset_train, dataset_val, X_normalized, X = artifical_GP_generation(n_observations_per_group = 1000,
                                                n_dimensions = 10,
                                                n_time_points = n_time_points,
                                                n_new_dims = n_new_dims,
                                                p_dataset = 0.5)

from pypots.utils.metrics import calc_mae
from pypots.optim.adam import Adam
from torch.optim.lr_scheduler import LRScheduler

from pypots.imputation import GP_VAE

gpvae = GP_VAE(n_steps = dataset_train['X'].shape[1], 
            n_features = dataset_train['X'].shape[2], 
            latent_size = 10, 
            epochs = 300, 
            batch_size = 32,
            beta = 1., 
            K = 5,  
            encoder_sizes = (128,128), 
            decoder_sizes = (256,256),
            optimizer = Adam(weight_decay=0.001), #lr = 1e-4
            patience = 30
            )

gpvae.model.backbone.alpha = 1e0
gpvae.model.backbone.beta = 1e-2
gpvae.model.backbone.gamma = 5 * 1e-5
gpvae.model.backbone.sigma1 = 1e-3
gpvae.model.backbone.sigma2 = 1e-2

gpvae.model.backbone.sampling = True
gpvae.model.backbone.use_mean = True
gpvae.model.backbone.noise_std = 1e-2
gpvae.model.backbone.detach = True
gpvae.model.backbone.compensate = False
gpvae.use_gp = False
gpvae.train_gp = False
gpvae.gp.plot_while_training = False

gpvae.model.backbone.p = 0.3
gpvae.model.backbone.device = 'cpu'

gpvae2 = GP_VAE(n_steps = dataset_train['X'].shape[1], 
            n_features = dataset_train['X'].shape[2], 
            latent_size = 10, 
            epochs = 300, 
            batch_size = 32,
            beta = 1., 
            K = 5,  
            encoder_sizes = (128,128), 
            decoder_sizes = (256,256),
            optimizer = Adam(weight_decay=0.001), #lr = 1e-4
            patience = 30
            )

gpvae2.model.backbone.alpha = 1e0
gpvae2.model.backbone.beta = 1e-2
gpvae2.model.backbone.gamma = 5 * 1e-5
gpvae2.model.backbone.sigma1 = 1e-3
gpvae2.model.backbone.sigma2 = 1e-1

gpvae2.model.backbone.sampling = True
gpvae2.model.backbone.use_mean = True
gpvae2.model.backbone.noise_std = 1e-2
gpvae2.model.backbone.detach = True
gpvae2.model.backbone.compensate = False
gpvae2.use_gp = False
gpvae2.train_gp = False
gpvae2.gp.plot_while_training = False

gpvae2.model.backbone.p = 0.3
gpvae2.model.backbone.device = 'cpu'


from pypots.optim import Adam
from pypots.imputation import SAITS

# initialize the model
saits = SAITS(
    n_steps=n_time_points,
    n_features=n_new_dims,
    n_layers=2,
    d_model=256,
    d_ffn=128,
    n_heads=4,
    d_k=64,
    d_v=64,
    dropout=0.1,
    attn_dropout=0.1,
    diagonal_attention_mask=True,  # otherwise the original self-attention mechanism will be applied
    ORT_weight=1,  # you can adjust the weight values of arguments ORT_weight
    # and MIT_weight to make the SAITS model focus more on one task. Usually you can just leave them to the default values, i.e. 1.
    MIT_weight=1,
    batch_size=32,
    # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
    epochs=300,
    # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
    # You can leave it to defualt as None to disable early stopping.
    patience=30,
    # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
    # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
    optimizer=Adam(lr=1e-3),
    # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
    # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
    # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
    num_workers=0,
    # just leave it to default as None, PyPOTS will automatically assign the best device for you.
    # Set it as 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices, even parallelly on ['cuda:0', 'cuda:1']
    device=None,  
    # set the path for saving tensorboard and trained model files 
    saving_path="tutorial_results/imputation/saits",
    # only save the best model after training finished.
    # You can also set it as "better" to save models performing better ever during training.
    model_saving_strategy="best",
)

from pypots.imputation import GPVAE

gp_vae = GPVAE(
    n_steps=n_time_points,
    n_features=n_new_dims,
    latent_size=37,
    encoder_sizes = (128,128), 
    decoder_sizes = (256,256),
    optimizer = Adam(weight_decay=0.001), #lr = 1e-4
    kernel="cauchy",
    beta=0.2,
    M=1,
    K=1,
    sigma=1.005,
    length_scale=7.0,
    kernel_scales=1,
    window_size=24,
    batch_size=32,
    # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
    epochs=300,
    # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
    # You can leave it to default as None to disable early stopping.
    patience=5,
    # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
    # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
    # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
    # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
    # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
    num_workers=0,
    # just leave it to default as None, PyPOTS will automatically assign the best device for you.
    # Set it as 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices, even parallelly on ['cuda:0', 'cuda:1']
    device=None,
    # set the path for saving tensorboard and trained model files 
    saving_path="tutorial_results/imputation/gp_vae",
    # only save the best model after training finished.
    # You can also set it as "better" to save models performing better ever during training.
    model_saving_strategy="best",
)

from pypots.imputation import CSDI

# initialize the model
csdi = CSDI(
    n_steps=n_time_points,
    n_features=n_new_dims,
    n_layers=6,
    n_heads=2,
    n_channels=128,
    d_time_embedding=64,
    d_feature_embedding=32,
    d_diffusion_embedding=128,
    target_strategy="random",
    n_diffusion_steps=50,
    batch_size=32,
    # here we set epochs=10 for a quick demo, you can set it to 100 or more for better performance
    epochs=300,
    # here we set patience=3 to early stop the training if the evaluting loss doesn't decrease for 3 epoches.
    # You can leave it to defualt as None to disable early stopping.
    patience=5,
    # give the optimizer. Different from torch.optim.Optimizer, you don't have to specify model's parameters when
    # initializing pypots.optim.Optimizer. You can also leave it to default. It will initilize an Adam optimizer with lr=0.001.
    optimizer=Adam(lr=1e-3),
    # this num_workers argument is for torch.utils.data.Dataloader. It's the number of subprocesses to use for data loading.
    # Leaving it to default as 0 means data loading will be in the main process, i.e. there won't be subprocesses.
    # You can increase it to >1 if you think your dataloading is a bottleneck to your model training speed
    num_workers=0,
    # just leave it to default as None, PyPOTS will automatically assign the best device for you.
    # Set it as 'cpu' if you don't have CUDA devices. You can also set it to 'cuda:0' or 'cuda:1' if you have multiple CUDA devices, even parallelly on ['cuda:0', 'cuda:1']
    device=None,
    # set the path for saving tensorboard and trained model files 
    saving_path="tutorial_results/imputation/csdi",
    # only save the best model after training finished.
    # You can also set it as "better" to save models performing better ever during training.
    model_saving_strategy="best",
)

models = [saits, gpvae, gp_vae, saits]
models = {'gpvae2':gpvae2,
            'saits':saits,
            'gpvae':gpvae,
            'gp_vae':gp_vae
            } #,'gp_vae':gp_vae,'saits':saits

from pypots.utils.metrics import calc_mae

scores = {}

for model_name in models.keys():
    model = models[model_name]
    model.fit(dataset_train_nan, dataset_val_nan)
    try:
        results = model.predict(dataset_val_nan, with_gp = True)
    except:
        results = model.predict(dataset_val_nan)
    imputation = results['imputation']
    if imputation.shape[1] != dataset_val_nan['X_ori'].shape[1]:
        imputation = imputation.reshape(dataset_val_nan['X_ori'].shape)
    testing_mae = calc_mae(imputation, dataset_val_nan['X_ori'], (dataset_val_nan['X_ori']==dataset_val_nan['X_ori']))
    scores[model_name] = testing_mae
    print(scores)

print(scores)


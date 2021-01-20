# import modules
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from models.cond_GAN import cond_GAN
from keras.utils import plot_model

# 0. set parameters for the runs/studies
SECTION = 'cond_gan'
RUN_ID = '0001'
DATA_NAME = 'cell'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER,'viz'))
    os.mkdir(os.path.join(RUN_FOLDER,'images'))
    os.mkdir(os.path.join(RUN_FOLDER,'weights'))
    os.mkdir(os.path.join(RUN_FOLDER,'plots'))

mode = 'build'
# mode = 'load'
# 1. import the data
data = sio.loadmat('data/unit_cells_all_sca_rec_ell_result.mat')
X_pix = data['X_pixel']
Y = data['Y']
freq = data['freq']
# Data scaling: Remember to scale the data to the range between -1 and 1! 
scaler = StandardScaler()
# scaler = MinMaxScaler((-1,1))
# pca = PCA(n_components=0.95)
# Y = pca.fit_transform(Y)
Y_scaled = scaler.fit_transform(Y)

n_tl_points = np.shape(Y_scaled)[1]
n_samples = np.shape(X_pix)[0]
X_pix = np.reshape(X_pix, (n_samples, 88, 88, 1))


X_train, X_test, Y_train, Y_test = train_test_split(X_pix, Y_scaled, test_size=0.15, random_state=42)



IMAGE_SIZE = 88

cgan = cond_GAN(input_dim = (IMAGE_SIZE,IMAGE_SIZE,1)
        , tl_dim = n_tl_points
        , input_data = X_train
        , tl_data = Y_train
        , discriminator_conv_filters = [32,32,64,64]
        , discriminator_conv_kernel_size = [5,5,5,5]
        , discriminator_conv_strides = [2,2,2,1]
        , discriminator_batch_norm_momentum = None
        , discriminator_activation = 'leaky_relu'
        , discriminator_dropout_rate = 0.4
        , discriminator_learning_rate = 0.0008
        , generator_initial_dense_layer_size = (11, 11, 64)
        , generator_upsample = [2,2,2,1]
        , generator_conv_filters = [128,32,32,1]
        , generator_conv_kernel_size = [5,5,5,5]
        , generator_conv_strides = [1,1,1,1]
        , generator_batch_norm_momentum = 0.9
        , generator_activation = 'leaky_relu'
        , generator_dropout_rate = None
        , generator_learning_rate = 0.0004
        , optimiser = 'adam'
        , z_dim = 128
        )

cgan.discriminator.summary()
cgan.generator.summary()
plot_model(cgan.discriminator, to_file='disc.png', show_shapes=False, show_layer_names=True)
plot_model(cgan.generator, to_file='gen.png', show_shapes=False, show_layer_names=True)


# 3. training phase
# set the training parameters
EPOCHS = 2000
PRINT_EVERY_N_BATCHES = 200
BATCH_SIZE = 32

# launch the training process
cgan.train(
    X_train
    , batch_size = BATCH_SIZE
    , epochs = EPOCHS
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
    , using_generator = False
)

# plot: loss over epoch for generator and for critic
#wgan.sample_images(RUN_FOLDER)

fig = plt.figure()
plt.plot([x[0] for x in cgan.d_losses], color='black', linewidth=1.0)
plt.plot([x[1] for x in cgan.d_losses], color='green', linewidth=1.0)
plt.plot([x[2] for x in cgan.d_losses], color='red', linewidth=1.0)
plt.plot([x[0] for x in cgan.g_losses], color='orange', linewidth=1.0)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.xlim(0, EPOCHS)
fig.savefig(os.path.join(RUN_FOLDER, "plots/loss_epoch.pdf"))

# plt.close()

fig2 = plt.figure()
plt.plot([x[3] for x in cgan.d_losses], color='black', linewidth=1.0)
plt.plot([x[4] for x in cgan.d_losses], color='green', linewidth=1.0)
plt.plot([x[5] for x in cgan.d_losses], color='red', linewidth=1.0)
plt.plot([x[1] for x in cgan.g_losses], color='orange', linewidth=1.0)
plt.xlabel('epoch', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.xlim(0, EPOCHS)
fig2.savefig(os.path.join(RUN_FOLDER, "plots/accuracy_epoch.pdf"))

mdic = {"d_losses": cgan.d_losses, "g_losses": cgan.g_losses}
sio.savemat("losses_gan.mat", mdic)

# test (true) images from the training data set
r, c = 3, 3
idx = np.random.randint(0, X_train.shape[0], BATCH_SIZE)
true_imgs = (X_train[idx] + 1) *0.5

fig, axs = plt.subplots(r, c, figsize=(9,9))
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i,j].imshow(np.squeeze(true_imgs[cnt, :, :, :]), cmap = 'gray')
        axs[i,j].axis('off')
        cnt += 1
fig.savefig(os.path.join(RUN_FOLDER, "images/real.png"))
# plt.close()

# predictions of the generator, generated images
r, c = 3, 3
noise = np.random.normal(0, 1, (r * c, cgan.z_dim))
noise, tl_input = cgan.generate_latent_points(r*c)
gen_imgs = cgan.generator.predict([noise, tl_input])
# Rescale images 0 - 1
gen_imgs = 0.5 * (gen_imgs + 1)
# gen_imgs = np.clip(gen_imgs, 0, 1)
fig, axs = plt.subplots(r, c, figsize=(9, 9))
cnt = 0

for i in range(r):
    for j in range(c):
        axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap='gray')
        axs[i, j].axis('off')
        cnt += 1
fig.savefig(os.path.join(RUN_FOLDER, "images/sample_gan.png"))
# plt.close()
plt.show()


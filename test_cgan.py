import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from models.cond_GAN import cond_GAN
from keras.utils import plot_model
from keras.models import load_model
import datetime


# %% 0. set parameters for the runs/studies
SECTION = 'cond_gan'
RUN_ID = '0001'
DATA_NAME = 'cell'
RUN_FOLDER = 'run/{}/'.format(SECTION)
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'viz'))
    os.mkdir(os.path.join(RUN_FOLDER, 'images'))
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
    os.mkdir(os.path.join(RUN_FOLDER, 'plots'))

# mode = 'build'
mode = 'load'

# %% 1. import the data
data = sio.loadmat('data/unit_cells_all_sca_rec_ell_result.mat')
X_pix = data['X_pixel']
Y = data['Y']
freq = data['freq']
# Data scaling: Remember to scale the data to the range between -1 and 1!
scaler = StandardScaler()
Y_scaled = scaler.fit_transform(Y)

n_tl_points = np.shape(Y_scaled)[1]
n_samples = np.shape(X_pix)[0]
x_train = np.reshape(X_pix, (n_samples, 88, 88, 1))

IMAGE_SIZE = 88
generator = load_model(os.path.join(RUN_FOLDER, 'generator.h5'))
generator.summary()

# %% 2. predictions of the generator, generated images

# choose 9 random sample images from dataset
ind_samples = np.random.randint(0, n_samples, (9,))

r, c = 3, 3  # plot grid
z_dim = 128  # latent space

tl_input = Y_scaled[ind_samples, :]  # select TLs to the 9 corresponding cells
noise = np.random.normal(0, 1, (r * c, 128))  # generate noise vector

# merge noise and tl vector
gen_input = [noise, tl_input]

# generate 9 img samples from the corresponding TL and noise inputs
gen_imgs = generator.predict(gen_input)

cnt = 0  # loop count

fig, axs = plt.subplots(r, c, figsize=(9, 9))
for i in range(r):
    for j in range(c):
        axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap='gray')
        axs[i, j].axis('off')
        cnt += 1

fig.savefig(os.path.join(RUN_FOLDER, "images/sample_gan.png"))
# plt.close()
plt.show()

cnt = 0  # loop count

# show the 9 real images
fig, axs = plt.subplots(r, c, figsize=(9, 9))
for i in range(r):
    for j in range(c):
        axs[i, j].imshow(np.squeeze(X_pix[ind_samples[cnt], :, :]), cmap='gray')
        axs[i, j].axis('off')
        cnt += 1


# %% image processing (light)

# convert img to binary img
def convert2binary(img, threshold):
    n_i = img.shape[0]
    n_j = img.shape[1]

    bin_img = img

    # loop over rows and cols
    for i in range(n_i):
        for j in range(n_j):

            # if pixel value >= threshold -> bin_img = 1
            if (img[i, j] >= threshold):
                bin_img[i, j] = 1

            # else: pixel value < threshild -> bin_img = -1
            else:
                bin_img[i, j] = -1

    return bin_img


# extract sample img
img_sample = np.squeeze(gen_imgs[0, :, :, :])

# convert sample img 2 binary image
bin_img_sample = convert2binary(img_sample, 0)

# plot sample img (left) and binary img (right)
fig, axs = plt.subplots(1, 2)
axs[0].imshow(img_sample, cmap='gray')
axs[0].axis('off')
axs[1].imshow(bin_img_sample, cmap='gray')
axs[1].axis('off')

# binary image stack
bin_img_stack = np.zeros((r * c, IMAGE_SIZE, IMAGE_SIZE))

# convert all 9 sample images
cnt = 0

fig, axs = plt.subplots(r, c, figsize=(9, 9))
for i in range(r):
    for j in range(c):
        # convert img 2 binary image
        bin_img_stack[cnt, :, :] = convert2binary(np.squeeze(gen_imgs[cnt, :, :, :]), 0)

        # plot
        axs[i, j].imshow(np.squeeze(bin_img_stack[cnt, :, :]), cmap='gray')
        axs[i, j].axis('off')
        cnt += 1


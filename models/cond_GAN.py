
from keras.layers import Input, Conv2D, Concatenate, Flatten, Dense, Conv2DTranspose, Reshape, Lambda, Activation, BatchNormalization, LeakyReLU, Dropout, ZeroPadding2D, UpSampling2D
from keras.layers.merge import _Merge

from keras.models import Model, Sequential
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras.initializers import RandomNormal

import numpy as np
import json
import os
import pickle as pkl
import matplotlib.pyplot as plt


class cond_GAN():
    def __init__(self
        , input_dim
        , tl_dim
        , input_data
        , tl_data
        , discriminator_conv_filters
        , discriminator_conv_kernel_size
        , discriminator_conv_strides
        , discriminator_batch_norm_momentum
        , discriminator_activation
        , discriminator_dropout_rate
        , discriminator_learning_rate
        , generator_initial_dense_layer_size
        , generator_upsample
        , generator_conv_filters
        , generator_conv_kernel_size
        , generator_conv_strides
        , generator_batch_norm_momentum
        , generator_activation
        , generator_dropout_rate
        , generator_learning_rate
        , optimiser
        , z_dim
        ):

        self.name = 'gan'

        self.input_dim = input_dim
        self.tl_dim = tl_dim
        self.input_data = input_data
        self.tl_data = tl_data
        self.discriminator_conv_filters = discriminator_conv_filters
        self.discriminator_conv_kernel_size = discriminator_conv_kernel_size
        self.discriminator_conv_strides = discriminator_conv_strides
        self.discriminator_batch_norm_momentum = discriminator_batch_norm_momentum
        self.discriminator_activation = discriminator_activation
        self.discriminator_dropout_rate = discriminator_dropout_rate
        self.discriminator_learning_rate = discriminator_learning_rate

        self.generator_initial_dense_layer_size = generator_initial_dense_layer_size
        self.generator_upsample = generator_upsample
        self.generator_conv_filters = generator_conv_filters
        self.generator_conv_kernel_size = generator_conv_kernel_size
        self.generator_conv_strides = generator_conv_strides
        self.generator_batch_norm_momentum = generator_batch_norm_momentum
        self.generator_activation = generator_activation
        self.generator_dropout_rate = generator_dropout_rate
        self.generator_learning_rate = generator_learning_rate
        
        self.optimiser = optimiser
        self.z_dim = z_dim

        self.n_layers_discriminator = len(discriminator_conv_filters)
        self.n_layers_generator = len(generator_conv_filters)

        self.weight_init = RandomNormal(mean=0., stddev=0.02)

        self.d_losses = []
        self.g_losses = []

        self.epoch = 0
        self.cnt = 0

        self._build_discriminator()
        self._build_generator()

        self._build_adversarial()

    def get_activation(self, activation):
        if activation == 'leaky_relu':
            layer = LeakyReLU(alpha = 0.2)
        else:
            layer = Activation(activation)
        return layer

    def _build_discriminator(self):

        ### THE discriminator
        # input channel for the transmission loss
        tl_input = Input(shape=(self.tl_dim,), name='discr_tl_input')
        n_neurons = self.input_dim[0]*self.input_dim[1]
        li = Dense(n_neurons)(tl_input)
        li = Reshape((self.input_dim[0], self.input_dim[1], 1))(li)

        image_input = Input(shape=self.input_dim, name='discri_image_input')


        discriminator_input = Concatenate()([image_input, li])

        x = discriminator_input

        for i in range(self.n_layers_discriminator):

            x = Conv2D(
                filters = self.discriminator_conv_filters[i]
                , kernel_size = self.discriminator_conv_kernel_size[i]
                , strides = self.discriminator_conv_strides[i]
                , padding = 'same'
                , name = 'discriminator_conv_' + str(i)
                , kernel_initializer = self.weight_init
                )(x)

            if self.discriminator_batch_norm_momentum and i > 0:
                x = BatchNormalization(momentum = self.discriminator_batch_norm_momentum)(x)

            x = self.get_activation(self.discriminator_activation)(x)

            if self.discriminator_dropout_rate:
                x = Dropout(rate = self.discriminator_dropout_rate)(x)

        x = Flatten()(x)
        
        discriminator_output = Dense(1, activation='sigmoid', kernel_initializer = self.weight_init)(x)

        self.discriminator = Model([image_input, tl_input], discriminator_output)


    def _build_generator(self):

        ### THE generator
        # generator input channel for the transmission loss
        tl_input = Input(shape=(self.tl_dim,), name='generator_tl_input')
        n_neurons = 11 * 11
        li = Dense(n_neurons)(tl_input)
        li = Reshape((11, 11, 1))(li)
        # generator input channel
        generator_input = Input(shape=(self.z_dim,), name='generator_input')

        x = generator_input

        x = Dense(np.prod(self.generator_initial_dense_layer_size), kernel_initializer = self.weight_init)(x)

        if self.generator_batch_norm_momentum:
            x = BatchNormalization(momentum = self.generator_batch_norm_momentum)(x)

        x = self.get_activation(self.generator_activation)(x)

        x = Reshape(self.generator_initial_dense_layer_size)(x)

        if self.generator_dropout_rate:
            x = Dropout(rate = self.generator_dropout_rate)(x)

        # merge the 2 input channels into one
        merged_input = Concatenate(name='generator_merged_input')([x, li])

        for i in range(self.n_layers_generator):

            if self.generator_upsample[i] == 2:
                merged_input = UpSampling2D()(merged_input)
                merged_input = Conv2D(
                    filters = self.generator_conv_filters[i]
                    , kernel_size = self.generator_conv_kernel_size[i]
                    , padding = 'same'
                    , name = 'generator_conv_' + str(i)
                    , kernel_initializer = self.weight_init
                )(merged_input)
            else:

                merged_input = Conv2DTranspose(
                    filters = self.generator_conv_filters[i]
                    , kernel_size = self.generator_conv_kernel_size[i]
                    , padding = 'same'
                    , strides = self.generator_conv_strides[i]
                    , name = 'generator_conv_' + str(i)
                    , kernel_initializer = self.weight_init
                    )(merged_input)

            if i < self.n_layers_generator - 1:

                if self.generator_batch_norm_momentum:
                    merged_input = BatchNormalization(momentum = self.generator_batch_norm_momentum)(merged_input)

                merged_input = self.get_activation(self.generator_activation)(merged_input)
                    
                
            else:

                merged_input = Activation('tanh')(merged_input)


        generator_output = merged_input

        self.generator = Model([generator_input, tl_input], generator_output)

       
    def get_opti(self, lr):
        if self.optimiser == 'adam':
            opti = Adam(lr=lr, beta_1=0.5)
        elif self.optimiser == 'rmsprop':
            opti = RMSprop(lr=lr)
        else:
            opti = Adam(lr=lr)

        return opti

    def set_trainable(self, m, val):
        m.trainable = val
        for l in m.layers:
            l.trainable = val


    def _build_adversarial(self):
        
        ### COMPILE DISCRIMINATOR

        self.discriminator.compile(
        optimizer=self.get_opti(self.discriminator_learning_rate)  
        , loss = 'binary_crossentropy'
        ,  metrics = ['accuracy']
        )
        
        ### COMPILE THE FULL GAN

        self.set_trainable(self.discriminator, False)

        noise, tl = self.generator.input
        gen_img_output = self.generator.output
        gan_output = self.discriminator([gen_img_output, tl])
        self.model = Model([noise, tl], gan_output)

        # model_input = Input(shape=(self.z_dim,), name='model_input')
        # model_output = self.discriminator(self.generator(model_input))
        # self.model = Model(model_input, model_output)

        self.model.compile(optimizer=self.get_opti(self.generator_learning_rate) , loss='binary_crossentropy', metrics=['accuracy'])

        self.set_trainable(self.discriminator, True)

    def generate_real_samples(self, n_samples):
        # split into images and transmission loss
        images = self.input_data
        tl = self.tl_data
        # choose random instances from the dataset
        ix = np.random.randint(0, images.shape[0], n_samples)
        # select images and tl
        X, tl = images[ix], tl[ix]
        # generate class labels for 'real' classifications
        y = np.ones((n_samples, 1))
        return X, tl

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, generator, n_samples):
        # generate points in latent space
        z_input, tl_input = self.generate_latent_points(n_samples)
        # predict outputs
        images = generator.predict([z_input, tl_input])
        # create class labels for 'fake' classification
        y = - np.ones((n_samples, 1))
        return images, tl_input

    def generate_latent_points(self, n_samples):
        # generate points in the latent space
        # x_input = randn(self.z_dim * n_samples)
        # reshape into a batch of inputs for the network
        # z_input = x_input.reshape(n_samples, latent_dim)
        z_input = np.random.normal(0, 1, (n_samples, self.z_dim))
        # generate randomly selected transmission loss
        tl_ix = np.random.randint(0, self.tl_data.shape[0], n_samples)
        tl_input = self.tl_data[tl_ix, :]
        return [z_input, tl_input]

    
    def train_discriminator(self, x_train, batch_size, using_generator):

        valid = np.ones((batch_size,1))
        fake = np.zeros((batch_size,1))

        # if using_generator:
        #     true_imgs = next(x_train)[0]
        #     if true_imgs.shape[0] != batch_size:
        #         true_imgs = next(x_train)[0]
        # else:
        #     idx = np.random.randint(0, x_train.shape[0], batch_size)
        #     true_imgs = x_train[idx]
        #
        # noise = np.random.normal(0, 1, (batch_size, self.z_dim))
        # gen_imgs = self.generator.predict(noise)

        # generate real samples
        real_imgs, real_tl = self.generate_real_samples(batch_size)
        # generate fake samples
        fake_imgs, fake_tl = self.generate_fake_samples(self.generator, batch_size)

        d_loss_real, d_acc_real =   self.discriminator.train_on_batch([real_imgs, real_tl], valid)
        d_loss_fake, d_acc_fake =   self.discriminator.train_on_batch([fake_imgs, fake_tl], fake)
        d_loss =  0.5 * (d_loss_real + d_loss_fake)
        d_acc = 0.5 * (d_acc_real + d_acc_fake)

        return [d_loss, d_loss_real, d_loss_fake, d_acc, d_acc_real, d_acc_fake]
        # return [d_loss, d_loss_real, d_loss_fake]

    def train_generator(self, batch_size):
        valid = np.ones((batch_size, 1))
        noise, tl_input = self.generate_latent_points(batch_size)
        return self.model.train_on_batch([noise, tl_input], valid)


    def train(self, x_train, batch_size, epochs, run_folder
    , print_every_n_batches = 50
    , using_generator = False):
        #for morph plot
        # fig, axs = plt.subplots(3, 6, figsize=(6, 15))

        for epoch in range(self.epoch, self.epoch + epochs):

            d = self.train_discriminator(x_train, batch_size, using_generator)
            g = self.train_generator(batch_size)

            # print ("%d [D loss: (%.3f)(R %.3f, F %.3f)] [D acc: (%.3f)(%.3f, %.3f)] [G loss: %.3f] [G acc: %.3f]" % (epoch, d[0], d[1], d[2], d[3], d[4], d[5], g[0], g[1]))
            print("%d [D loss: (%.3f)(R %.3f, F %.3f)] [G loss: %.3f]" % (epoch, d[0], d[1], d[2], g[0]))

            self.d_losses.append(d)
            self.g_losses.append(g)

            if epoch % print_every_n_batches == 0:
                self.sample_images(run_folder)
                # self.sample_morphing_images(epoch, self.cnt, fig, axs, run_folder)
                # self.cnt += 1
                # self.model.save_weights(os.path.join(run_folder, 'weights/weights-%d.h5' % (epoch)))
                # self.model.save_weights(os.path.join(run_folder, 'weights/weights.h5'))
                self.save_model(run_folder)

            self.epoch += 1

    def sample_morphing_images(self, epoch, cnt, fig, axs, run_folder):
        r, c = 3, 6
        np.random.seed(42)
        noise, tl_input = self.generate_latent_points(r*c)
        # noise = np.random.normal(0, 1, (r*c, self.z_dim))
        gen_imgs = self.generator.predict([noise, tl_input])

        # rescales images 0-1
        gen_imgs = 0.5*(gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0 , 1)

        # fig, axs = plt.subplots(r, c, figsize=(15,15))

        for i in range(r):
            axs[i, self.cnt].imshow(np.squeeze(gen_imgs[i, :,:,:]), cmap='gray')
            axs[i, self.cnt].axis('off')


        plt.tight_layout()
        fig.savefig(os.path.join(run_folder, "images/sample_morphing_%d.png" % self.epoch))
        # plt.close()

    def sample_images(self, run_folder):
        r, c = 5, 5
        # noise = np.random.normal(0, 1, (r * c, self.z_dim))
        noise, tl_input = self.generate_latent_points(r * c)
        gen_imgs = self.generator.predict([noise, tl_input])

        # Rescale images 0 - 1

        gen_imgs = 0.5 * (gen_imgs + 1)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(r, c, figsize=(15, 15))
        cnt = 0

        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(np.squeeze(gen_imgs[cnt, :, :, :]), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(os.path.join(run_folder, "images/sample_%d.png" % self.epoch))
        plt.close()




    
    def plot_model(self, run_folder):
        plot_model(self.model, to_file=os.path.join(run_folder ,'viz/model.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.discriminator, to_file=os.path.join(run_folder ,'viz/discriminator.png'), show_shapes = True, show_layer_names = True)
        plot_model(self.generator, to_file=os.path.join(run_folder ,'viz/generator.png'), show_shapes = True, show_layer_names = True)



    def save(self, folder):

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pkl.dump([
                self.input_dim
                , self.discriminator_conv_filters
                , self.discriminator_conv_kernel_size
                , self.discriminator_conv_strides
                , self.discriminator_batch_norm_momentum
                , self.discriminator_activation
                , self.discriminator_dropout_rate
                , self.discriminator_learning_rate
                , self.generator_initial_dense_layer_size
                , self.generator_upsample
                , self.generator_conv_filters
                , self.generator_conv_kernel_size
                , self.generator_conv_strides
                , self.generator_batch_norm_momentum
                , self.generator_activation
                , self.generator_dropout_rate
                , self.generator_learning_rate
                , self.optimiser
                , self.z_dim
                ], f)

        self.plot_model(folder)

    def save_model(self, run_folder):
        self.model.save(os.path.join(run_folder, 'model.h5'))
        self.discriminator.save(os.path.join(run_folder, 'discriminator.h5'))
        self.generator.save(os.path.join(run_folder, 'generator.h5'))
        # pkl.dump(self, open( os.path.join(run_folder, "obj.pkl"), "wb" ))

    def load_weights(self, filepath):
        self.model.load_weights(filepath)


        



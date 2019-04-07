import os
from   tensorflow.keras.preprocessing import image
 
import tensorflow.keras
from   tensorflow.keras import layers
import numpy as np


latent_dim = 32


height, width, channels = 32, 32, 3

##  Generator.
generator_input = tensorflow.keras.Input(shape=(latent_dim,))
x = layers.Dense(128 * 16 * 16)(generator_input)
x = layers.LeakyReLU()(x)
x = layers.Reshape((16, 16, 128))(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2DTranspose(256, 4, strides=2, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(256, 5, padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(channels, 7, activation='tanh', padding='same')(x)
generator = tensorflow.keras.models.Model(generator_input, x)
generator.summary()


##  Discriminator.
discriminator_input = layers.Input(shape=(height, width, channels))
x = layers.Conv2D(128, 3)(discriminator_input)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)   
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Conv2D(128, 4, strides=2)(x)
x = layers.LeakyReLU()(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(1, activation='sigmoid')(x)
discriminator = tensorflow.keras.models.Model(discriminator_input, x)
discriminator.summary()

discriminator_optimizer = tensorflow.keras.optimizers.RMSprop(
    lr=0.0008,
    clipvalue=1.0,
    decay=1e-8)

discriminator.compile(optimizer=discriminator_optimizer,
                     loss='binary_crossentropy')

##  GAN.
discriminator.trainable = False
gan_input = tensorflow.keras.Input(shape=(latent_dim,)) 
gan_output = discriminator(generator(gan_input)) 
gan = tensorflow.keras.models.Model(gan_input, gan_output)
gan_optimizer = tensorflow.keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8) 
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')


##  Run.
(x_train, y_train), (_, _) = tensorflow.keras.datasets.cifar10.load_data()

x_train = x_train[y_train.flatten() == 6]

x_train = x_train.reshape((x_train.shape[0],) + (height, width, channels)).astype('float32') / 255.

iterations = 10000
batch_size = 20
save_dir   = 'GANIM'

start = 0

for step in range(iterations):
    random_latent_vectors = np.random.normal(size=(batch_size, latent_dim))

    ##  Decodes to fake images. 
    generated_images = generator.predict(random_latent_vectors)

    stop             = start + batch_size
    real_images      = x_train[start: stop]

    ##  Combine with real images. 
    combined_images  = np.concatenate([generated_images, real_images])

    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

    ##  Important:  Add random noise to the labels. 
    labels += 0.05 * np.random.random(labels.shape)

    ##  Train the discriminator to predict real from fake. 
    d_loss = discriminator.train_on_batch(combined_images, labels)

    ##  Assembles labels lying these are all real. 
    misleading_targets = np.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

    start += batch_size

    if start > len(x_train) - batch_size:
        start = 0

    if step % 100 == 0:
        ##  Save and plot every 100.
        gan.save_weights('gan.h5')

        print('discriminator loss:', d_loss)
        print('adversarial loss:', a_loss)

        img = image.array_to_img(generated_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'generated_frog' + str(step) + '.png'))
        
        img = image.array_to_img(real_images[0] * 255., scale=False)
        img.save(os.path.join(save_dir, 'real_frog' + str(step) + '.png'))











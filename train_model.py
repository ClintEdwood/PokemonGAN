import os
# Disable info and warnings from TF
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Conv2DTranspose, Conv2D, BatchNormalization, LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import load_img, img_to_array
import numpy as np

data_path = "resizedData/"

# Define the image dimensions
img_width = 64
img_height = 64

# Load the images into a numpy array
image_list = []
for filename in os.listdir(data_path):
    img = load_img(os.path.join(data_path, filename), target_size=(img_width, img_height))
    img_array = img_to_array(img)
    image_list.append(img_array)

images = np.array(image_list)

# Normalize the pixel values to be between -1 and 1
x_train = (images.astype('float32') - 127.5) / 127.5

# Generator
generator = Sequential([
    Dense(256 * 8 * 8, input_dim=100),
    Reshape((8, 8, 256)),
    BatchNormalization(),
    Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'),
    LeakyReLU(alpha=0.2),
    BatchNormalization(),
    Conv2DTranspose(64, kernel_size=4, strides=2, padding='same'),
    LeakyReLU(alpha=0.2),
    Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')
])

# Discriminator
discriminator = Sequential([
    Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=(64, 64, 3)),
    LeakyReLU(alpha=0.2),
    Dropout(0.4),
    Conv2D(128, kernel_size=4, strides=2, padding='same'),
    LeakyReLU(alpha=0.2),
    Dropout(0.4),
    Conv2D(256, kernel_size=4, strides=2, padding='same'),
    LeakyReLU(alpha=0.2),
    Dropout(0.4),
    Flatten(),
    Dense(1, activation='sigmoid')
])

# Compile the models
generator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Combine generator and discriminator
inp = Input(shape=(100,))
image = generator(inp)
print(image)
validity = discriminator(image)
combined = Model(inp, validity)
discriminator.trainable = False
combined.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Train the model
batch_size = 128
epochs = 3000
for epoch in range(epochs):
    # Train the discriminator
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images = x_train[idx]
    noise = np.random.normal(0, 1, (batch_size, 100))
    fake_images = generator.predict(noise)
    discriminator_loss_real = discriminator.train_on_batch(real_images, np.ones((batch_size, 1)))
    discriminator_loss_fake = discriminator.train_on_batch(fake_images, np.zeros((batch_size, 1)))
    discriminator_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    generator_loss = combined.train_on_batch(noise, np.ones((batch_size, 1)))
    # Print the losses
    print(f'Epoch {epoch}: D loss = {discriminator_loss}, G loss = {generator_loss}')

generator.save("3000_generator")

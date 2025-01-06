import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import datasets
import tensorflow as tf

# Load dataset path and preprocess
DATASET_PATH = "path_to_downloaded_dataset"
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32

# Load and visualize the dataset
def load_images(data_path):
    images = []
    labels = []
    for folder in os.listdir(data_path):
        label = folder
        folder_path = os.path.join(data_path, folder)
        for image_file in os.listdir(folder_path):
            image = tf.keras.utils.load_img(
                os.path.join(folder_path, image_file),
                target_size=IMAGE_SIZE
            )
            image = tf.keras.utils.img_to_array(image)
            images.append(image)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load data
images, labels = load_images(DATASET_PATH)

# Encode labels and split data
label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
encoded_labels = np.array([label_mapping[label] for label in labels])
x_train, x_test, y_train, y_test = train_test_split(images / 255.0, encoded_labels, test_size=0.2, random_state=42)

# Data visualization
def visualize_data(images, labels, label_mapping):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i])
        plt.title(list(label_mapping.keys())[list(label_mapping.values()).index(labels[i])])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualize_data(x_train, y_train, label_mapping)

# Generator architecture for GAN
def build_generator():
    model = Sequential()
    model.add(Dense(256, activation=LeakyReLU(0.2), input_dim=100))
    model.add(BatchNormalization())
    model.add(Dense(512, activation=LeakyReLU(0.2)))
    model.add(BatchNormalization())
    model.add(Dense(1024, activation=LeakyReLU(0.2)))
    model.add(BatchNormalization())
    model.add(Dense(np.prod(IMAGE_SIZE) * 3, activation='tanh'))
    model.add(Reshape((*IMAGE_SIZE, 3)))
    return model

# Discriminator architecture for GAN
def build_discriminator():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=(*IMAGE_SIZE, 3), padding='same'))
    model.add(LeakyReLU(0.2))
    model.add(Conv2D(128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Build GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    generated_image = generator(gan_input)
    gan_output = discriminator(generated_image)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))
    return gan

# Create models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])
gan = build_gan(generator, discriminator)

# GAN training
def train_gan(generator, discriminator, gan, epochs, batch_size):
    noise_dim = 100
    real_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train discriminator
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[idx]
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        fake_images = generator.predict(noise)
        
        d_loss_real = discriminator.train_on_batch(real_images, real_labels)
        d_loss_fake = discriminator.train_on_batch(fake_images, fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, noise_dim))
        g_loss = gan.train_on_batch(noise, real_labels)

        # Print losses
        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} | D Loss: {d_loss[0]} | G Loss: {g_loss}")

# Train GAN
train_gan(generator, discriminator, gan, epochs=10000, batch_size=32)

# CNN for classification
cnn_model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*IMAGE_SIZE, 3)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(label_mapping), activation='softmax')
])

cnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train CNN
cnn_model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32)

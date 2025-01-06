import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Reshape, LeakyReLU, BatchNormalization, Input, UpSampling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# 1. Data Loading and Visualization
def load_dataset():
    # Replace with your dataset loading code
    # Example for loading images (assumes a folder structure for classes)
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    img_size = 64  # Image size for resizing
    batch_size = 32

    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_data = datagen.flow_from_directory(
        "./dataset",  # Path to your dataset folder
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='training'
    )

    val_data = datagen.flow_from_directory(
        "./dataset",
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_data, val_data, img_size

train_data, val_data, img_size = load_dataset()

# Plot some samples from the dataset
x_batch, y_batch = next(train_data)
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_batch[i])
    plt.axis('off')
plt.show()

# 2. GAN Architecture
# Generator
def build_generator(latent_dim, img_size):
    model = Sequential()

    model.add(Dense(128 * (img_size // 4) * (img_size // 4), input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape(((img_size // 4), (img_size // 4), 128)))

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(3, kernel_size=4, padding="same", activation='tanh'))

    return model

# Discriminator
def build_discriminator(img_size):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=4, strides=2, input_shape=(img_size, img_size, 3), padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=4, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    return model

# GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential([generator, discriminator])
    return model

latent_dim = 100
generator = build_generator(latent_dim, img_size)
discriminator = build_discriminator(img_size)
optimizer = Adam(0.0002, 0.5)
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Compile GAN
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

# 3. Training GAN
def train_gan(epochs, batch_size, train_data):
    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # Train discriminator
        imgs, _ = next(train_data)
        idx = np.random.randint(0, imgs.shape[0], batch_size)
        real_imgs = imgs[idx]

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(real_imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train generator
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        g_loss = gan.train_on_batch(noise, real)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D Loss: {d_loss[0]}, D Acc: {100 * d_loss[1]}, G Loss: {g_loss}")

            # Save generated images
            save_generated_images(epoch, generator, latent_dim)

# Save generated images
def save_generated_images(epoch, generator, latent_dim, examples=25):
    noise = np.random.normal(0, 1, (examples, latent_dim))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5

    plt.figure(figsize=(5, 5))
    for i in range(examples):
        plt.subplot(5, 5, i + 1)
        plt.imshow(gen_imgs[i])
        plt.axis('off')
    plt.savefig(f"gan_images_epoch_{epoch}.png")
    plt.close()

# Train the GAN
train_gan(epochs=5000, batch_size=32, train_data=train_data)

# 4. CNN/CGAN for Classification
# CNN model for classification
def build_cnn(img_size):
    model = Sequential([
        Conv2D(32, kernel_size=3, activation='relu', input_shape=(img_size, img_size, 3)),
        BatchNormalization(),
        Conv2D(64, kernel_size=3, activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

cnn = build_cnn(img_size)
history = cnn.fit(train_data, validation_data=val_data, epochs=10)

# Visualize Training Results
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

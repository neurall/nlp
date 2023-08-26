from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Autoencoder architecture
input_layer = Input(shape=(28, 28))
flatten = Flatten()(input_layer)
encoder_layer = Dense(128, activation='relu')(flatten)
encoder_layer = Dense(64, activation='relu')(encoder_layer)
encoder_layer = Dense(32, activation='relu')(encoder_layer)

decoder_layer = Dense(64, activation='relu')(encoder_layer)
decoder_layer = Dense(128, activation='relu')(decoder_layer)
decoder_layer = Dense(784, activation='sigmoid')(decoder_layer)
output_layer = Reshape((28, 28))(decoder_layer)

# Compile the autoencoder model
autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# Encode and decode some test images
decoded_imgs = autoencoder.predict(x_test)

# Display original and reconstructed images
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    # Original images
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i])
    plt.title("Original")
    plt.gray()
    ax.axis('off')

    # Reconstructed images
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i])
    plt.title("Reconstructed")
    plt.gray()
    ax.axis('off')

plt.show()

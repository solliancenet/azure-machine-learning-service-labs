
import numpy as np
# We use Keras framework to build Neural networks
import keras
from keras import backend as K

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.utils.vis_utils import plot_model


# Verify we have a GPU available
# The output of the following should not be an empty array
# If you get an empty array back, it means no GPU was detected, which might mean you need to 
# uninstall keras/tensorflow/tensorflow-gpu and then reinstall tensorflow-gpu and keras
K.tensorflow_backend._get_available_gpus()

# We use Fashion mnist dataset
from keras.datasets import fashion_mnist

# We download and load the data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# Build the encoder
input_img = Input(shape=(28, 28, 1)) # This will be the input 👚

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded_feature_vector = MaxPooling2D((2, 2), padding='same', name='feature_vector')(x) #  <---- name your feature vector somehow

# at this point the representation is (4, 4, 8) i.e. 128-dimensional compressed feature vector


# Build the decoder
x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded_feature_vector)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded_output = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)


# The first model is autoencoder model, it takes the input image and results in a decoded image
autoencoder_model = Model(input_img, decoded_output)
# Compile the first model
autoencoder_model.compile(optimizer='adadelta', loss='binary_crossentropy')


# The second NN model is only a half of the first model, it take the input image and gives the encoded vector as output
encoder_model = Model(inputs=autoencoder_model.input,
                                 outputs=autoencoder_model.get_layer('feature_vector').output) # <---- take the output from the feature vector
# Compile the second model
encoder_model.compile(optimizer='adadelta', loss='binary_crossentropy')

# We need to scale the image from [0-255] to [0-1] for better performance of activation functions
x_train = x_train / 255.
x_test = x_test / 255.


# We train the NN in batches (groups of images), so we reshape the dataset
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

print("Train dataset size is {0}".format(x_train.shape))
print("Test dataset size is {0}".format(x_test.shape))


# It takes several minutes to train this neural network, depending on the configuration of your cluster.
learning_history=autoencoder_model.fit(x=x_train, y=x_train, epochs=10, batch_size=128, 
                                 shuffle=True, validation_data=(x_test, x_test), verbose=1)


# Test the model
encoded_decoded_image=autoencoder_model.predict(x_test)


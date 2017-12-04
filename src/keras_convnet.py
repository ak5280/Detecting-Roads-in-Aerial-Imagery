from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import EarlyStopping

import numpy as np

def convnet():
    # In our case we will use a very small convnet with few layers and few
    # filters per layer, alongside data augmentation and dropout. Dropout also
    # helps reduce overfitting, by preventing a layer from seeing the exact same
    # pattern twice, thus acting in a way analoguous to data augmentation
    # (you could say that both dropout and data augmentation tend to disrupt
    # random correlations occuring in your data).

    # A simple stack of 3 convolution layers with a ReLU activation, followed by max-pooling layers
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # The model so far outputs 3D feature maps (height, width, features).

    # On top of it we stick two fully-connected layers.
    # We end the model with a single unit and a sigmoid activation, that outputs a
    # probability range between 0 and 1, which is perfect for a binary classification.
    # To go with it we will also use the binary_crossentropy loss to train our model.

    model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # In order to make the most of our few training examples, we will "augment" them
    # via a number of random transformations, so that our model would never see the
    # exact same picture twice. This helps prevent overfitting and helps generalize
    # the model better.

    # augmentation configuration used for training
    train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=90,
        shear_range=0.2,
        vertical_flip=True,
        horizontal_flip=True)

    # only rescaling used for testing:
    test_datagen = ImageDataGenerator(rescale=1/255)

    # this is the generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    # a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    early_stopping = EarlyStopping(patience=10)

    # we can now use these generators to train our model
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples//batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples//batch_size,
        callbacks=[early_stopping])

    # model.save_weights('weights_convnet.h5')
    # model.save('convnet.h5')

if __name__ == '__main__':
    # # fix random seed for reproducibility
    # np.random.seed(7)

    # dimensions of images
    img_width, img_height = 150, 150
    input_shape = (img_width, img_height, 3)

    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'
    nb_train_samples = 9063 #5797+3266
    nb_validation_samples = 2265 #1449+816
    epochs = 50
    batch_size = 16

    convnet()

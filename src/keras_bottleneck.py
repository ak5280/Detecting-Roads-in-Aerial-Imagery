from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
import numpy as np
import glob

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'

train_data_dir = 'data/train'
train_not_road_samples = len(glob.glob(train_data_dir+'/not_road/*.JPG'))
train_road_samples = len(glob.glob(train_data_dir+'/road/*.JPG'))
nb_train_samples = train_not_road_samples + train_road_samples # 9063 = 5797+3266 => not_road,road

validation_data_dir = 'data/validation'
validation_not_road_samples = len(glob.glob(validation_data_dir+'/not_road/*.JPG'))
validation_road_samples = len(glob.glob(validation_data_dir+'/road/*.JPG'))
nb_validation_samples = validation_not_road_samples + validation_road_samples # 2265 = 1449+816 => not_road,road

epochs = 50
batch_size = 16

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1/255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None, # our generator will only yield batches of data, no labels
        shuffle=False) # our data will be in order, so all first images will be "not_road", then "road"

    # the predict_generator method returns the output of a model,
    # given a generator that yields batches of numpy data
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples//batch_size)
    # save the output as a Numpy array
    np.save(open('bottleneck_features_train.npy', 'wb'),
        bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples//batch_size)
    np.save(open('bottleneck_features_validation.npy', 'wb'),
        bottleneck_features_validation)


def train_top_model():
    # we can then load our saved data and train a small fully-connected model
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    # the features were saved in order, so recreating the labels by each class subfolder
    train_labels = np.array([0]*train_not_road_samples +
        [1]*train_road_samples)

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    validation_labels = np.array([0]*validation_not_road_samples +
        [1]*validation_road_samples)

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    # thanks to its small size, this model trains very quickly even on CPU
    # so no need for early_stopping
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model()

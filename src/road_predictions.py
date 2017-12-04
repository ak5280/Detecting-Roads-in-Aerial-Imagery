
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import load_model

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import (accuracy_score, confusion_matrix,
                            classification_report, roc_curve, auc, roc_auc_score)
from sklearn import metrics

def model_acc_score():
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

    # evaluate the model
    scores = model.evaluate_generator(validation_generator, batch_size)
    print("Model %s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# A note on test and validation data:
# The Keras documentation uses three different sets of data: training data, validation data and test data.
# Training data is used to optimize the model parameters. The validation data is used to make choices about
# the meta-parameters, e.g. the number of epochs. After optimizing a model with optimal meta-parameters the
# test data is used to get a fair estimate of the model performance.

def boulder_results():
    boulder_test_gen = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(150, 150),
            batch_size=32,
            class_mode=('binary'),  # None => only data, no labels
            shuffle=False)  # False => keep data in same order as labels

    boulder_probabilities = model.predict_generator(boulder_test_gen, (1449+816)//batch_size)

    y_true_bd = np.array([0]*1449 + [1]*816)
    y_pred_bd = boulder_probabilities >= 0.5

    print("Accuracy score: ",accuracy_score(y_true_bd, y_pred_bd))

    print("Confusion matrix:\n",confusion_matrix(y_true_bd, y_pred_bd))

    print("Classification report:\n",classification_report(y_true_bd, y_pred_bd))

    print("AUC: ",roc_auc_score(y_true_bd, boulder_probabilities))

    # calculate the false positive rate and true positive rate for all thresholds of the classification
    fpr, tpr, thresholds = metrics.roc_curve(y_true_bd, boulder_probabilities)
    auc = metrics.auc(fpr, tpr)

    # ROC plot
    plt.title('Boulder - Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'darkorange', label = 'Area Under Curve = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def yuma_not_road_pred():
    yuma_not_road_path = '/Users/aa/Desktop/capstone/data/Imagery/tiles150/data/test_yuma/not_road/'
    yuma_not_road_img = np.random.choice(os.listdir(yuma_not_road_path))
    os.path.join(yuma_not_road_path, yuma_not_road_img)

    # keras load image to array
    img_ynr = load_img(os.path.join(yuma_not_road_path, yuma_not_road_img), target_size=(img_height, img_width))
    # normalize
    img_array_ynr = img_to_array(img_ynr)/255

    print(yuma_not_road_img)
    prob_ynr = model.predict(img_array_ynr.reshape(1,150,150,3))
    print("Probability Road: %s" % round(float(prob_ynr*100),2) + "%")

    plt.imshow(img_array_ynr)
    plt.suptitle(yuma_not_road_img)
    plt.title("Probability Road: %s" % round(float(prob_ynr*100),2) + "%")
    plt.show()

    # For VGG16
    # y_prob = model.predict(test)
    # pred_class_ynr = prob_ynr.argmax(axis=-1)
    # pred_class_ynr

    # For Sequential()
    pred_class_ynr = model.predict_classes(img_array_ynr.reshape(1,150,150,3))

    if pred_class_ynr == [[1]]:
        pred_class_ynr = "Road"
    else: pred_class_ynr = "Not Road"
    print("Predicted Class: %s" % pred_class_ynr)

def yuma_road_pred():
    yuma_road_path = '/Users/aa/Desktop/capstone/data/Imagery/tiles150/data/test_yuma/road/'
    yuma_road_img = np.random.choice(os.listdir(yuma_road_path))
    os.path.join(yuma_road_path, yuma_road_img)
    img_yr = load_img(os.path.join(yuma_road_path, yuma_road_img), target_size=(img_height, img_width))
    img_array_yr = img_to_array(img_yr)/255

    print(yuma_road_img)
    prob_yr = model.predict(img_array_yr.reshape(1,150,150,3))
    print("Probability Road: %s" % round(float(prob_yr*100),2) + "%")

    plt.imshow(img_array_yr)
    plt.suptitle(yuma_road_img)
    plt.title("Probability Road: %s" % round(float(prob_yr*100),2) + "%")
    plt.show()

    # For VGG16
    # pred_class_yr = prob_yr.argmax(axis=-1)
    # pred_class_yr

    # For Sequential()
    pred_class_yr = model.predict_classes(img_array_yr.reshape(1,150,150,3))

    if pred_class_yr == [[1]]:
        pred_class_yr = "Road"
    else: pred_class_yr = "Not Road"
    print("Predicted Class: %s" % pred_class_yr)

def yuma_results():
    # To get a confusion matrix from the test data you should go to two steps:
    #
    # Make predictions for the test data
    yuma_test_gen = test_datagen.flow_from_directory(
            'data/test_yuma',
            target_size=(150, 150),
            batch_size=batch_size,
            class_mode=('binary'),  # None => only data, no labels
            shuffle=False)  # False => keep data in same order as labels

    yuma_probabilities = model.predict_generator(yuma_test_gen, (8653+6387)//batch_size)

    # Compute the confusion matrix based on the label predictions
    y_true_yuma = np.array([0]*8653 + [1]*6387)
    y_pred_yuma = yuma_probabilities >= 0.5

    print("Accuracy score: ",accuracy_score(y_true_yuma, y_pred_yuma))

    print("Confusion matrix:\n",confusion_matrix(y_true_yuma, y_pred_yuma))

    print("Classification report:\n",classification_report(y_true_yuma, y_pred_yuma))

    print("AUC: ",roc_auc_score(y_true_yuma, yuma_probabilities))

    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, thresholds = metrics.roc_curve(y_true_yuma, yuma_probabilities)
    auc = metrics.auc(fpr, tpr)
    # ROC plot
    plt.title('Yuma - Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'darkorange', label = 'Area Under Curve = %0.2f' % auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == '__main__':
    # dimensions of images
    img_width, img_height = 150, 150
    input_shape = (img_width, img_height, 3)
    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'
    nb_train_samples = 9063 #5797+3266
    nb_validation_samples = 2265 #1449+816
    batch_size = 32

    model = load_model('convnet.h5')
    # print(model.summary())
    test_datagen = ImageDataGenerator(rescale=1/255)
    model_acc_score()
    boulder_results()
    yuma_not_road_pred()
    yuma_road_pred()
    yuma_results()

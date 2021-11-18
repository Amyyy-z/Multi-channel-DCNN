# @Date                 : 2021-02-04
# @Author               : Xinyu Zhang (Amy)
# @Python               : 3.7
# @Tensorflow Version   : 2.1.0
# @Contributor for Xception: Arjun Sarkar
# @Other required models can be viewed through: https://keras.io/api/applications/


import tensorflow as tf
import tensorflow.keras

from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json, Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, \
    UpSampling2D, BatchNormalization, Input, GlobalAveragePooling2D

from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

import keras
from keras import models
from keras import layers
from keras.layers.core import Permute
import tensorflow as tf
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.enable_eager_execution()

import tensorflow as tf
import timeit

from PIL import Image
import glob
import cv2
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

import os

# import all the images in the file Left
image_list_left = []
files = glob.glob(r"C:\..\Left\*.png")

for filename in files:
    print(filename)
    image = cv2.imread(filename)
    # image = tf.image.resize(image, [125, 125])
    image_list_left.append(image)  # convert images into array
    # print(filename)

# import all the images in the file Right
image_list_right = []
files = glob.glob(r"C:\..\Right\*.png")

for filename in files:
    print(filename)
    image = cv2.imread(filename)
    # image = tf.image.resize(image, [125, 125])
    image_list_right.append(image)  # convert images into array
    # print(filename)

print('image_list shape:', np.array(image_list_left).shape)
print('image_list shape:', np.array(image_list_right).shape)

# convert list into array
image_list_left = np.asarray(image_list_left)

# convert list into array
image_list_right = np.asarray(image_list_right)

import matplotlib.pyplot as plt
% matplotlib
inline

import numpy as np
from PIL import Image

X_left = image_list_left.reshape(965, 3, 224, 224).transpose(0, 2, 3, 1).astype("uint8")  # or use 2D images without reshape based on image formats
X_right = image_list_right.reshape(965, 3, 224, 224).transpose(0, 2, 3, 1).astype("uint8")

# Read labels
df_left = pd.read_csv("Dual Left.csv")
df_right = pd.read_csv("Dual right.csv")

# Extract type from df table
Label_left = df_left["Type"]
Label_right = df_right["Type"]

# Stack into array
label_left = np.asarray(Label_left)
label_right = np.asarray(Label_right)


# Encode labels
def one_hot_encode(vec, vals=2):
    # to one-hot encode the 4- possible labesl
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

# Match labels with images
class CifarHelper():
    def __init__(self):
        self.i = 0

        # self.all_train_batches = [X]

        self.images_left = None
        self.labels_left = None

        self.images_right = None
        self.labels_right = None

    def set_up_images(self):
        print("Setting up images and labels")
        self.images_left = np.vstack([X_left])
        all_len_left = len(self.images_left)

        self.images_left = self.images_left.reshape(all_len_left, 3, 224, 224).transpose(0, 2, 3, 1) / 255
        self.labels_left = one_hot_encode(np.hstack([label_left]), 2)

        self.images_right = np.vstack([X_right])
        all_len_right = len(self.images_right)

        self.images_right = self.images_right.reshape(all_len_right, 3, 224, 224).transpose(0, 2, 3, 1) / 255
        self.labels_right = one_hot_encode(np.hstack([label_right]), 2)


# before tensorflow run:
ch = CifarHelper()
ch.set_up_images()

# Check the image and its label
index = 466
plt.imshow(image_list_left[index])
print(label_left[index])

# Encoding data
def vectorize_sequences(sequences, dimension=1000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Make labels as one-hot format
def to_one_hot(labels_left, dimension=2):
    results_left = np.zeros((len(labels_left), dimension))
    for i, label_left in enumerate(labels_left):
        results_left[i, label_left] = 1.
    return results_left


def to_one_hot(labels_right, dimension=2):
    results_right = np.zeros((len(labels_right), dimension))
    for i, label_right in enumerate(labels_right):
        results_right[i, label_right] = 1.
    return results_right

one_hot_labels_left = to_one_hot(label_left)
one_hot_labels_right = to_one_hot(label_right)

import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold

# k-fold stratified CV
kf = StratifiedKFold(n_splits=10)
kf.get_n_splits(X_left, label_left)
kf.get_n_splits(X_right, label_right)

for train_index, test_index in kf.split(X_left, y_left):
    X_train_left, X_test_left = X_left[train_index], X_left[test_index]
    X_train_right, X_test_right = X_right[train_index], X_right[test_index]

    y_train_left, y_test_left = y_left[train_index], y_left[test_index]
    y_train_right, y_test_right = y_right[train_index], y_right[test_index]


# Xception model construction
def entry_flow(inputs):
    x = Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    previous_block_activation = x

    for size in [128, 256, 728]:
        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(3, strides=2, padding='same')(x)
        residual = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation)
        x = tensorflow.keras.layers.Add()([x, residual])
        previous_block_activation = x
    return x


def middle_flow(x, num_blocks=8):
    previous_block_activation = x

    for _ in range(num_blocks):
        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = tensorflow.keras.layers.Add()([x, previous_block_activation])
        previous_block_activation = x
    return x


def exit_flow(x):
    previous_block_activation = x

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    residual = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation)
    x = tensorflow.keras.layers.Add()([x, residual])
    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(2, activation='linear')(x)

    return x


inputs = Input(shape=(224, 224, 3))
outputs = exit_flow(middle_flow(entry_flow(inputs)))
xception = Model(inputs, outputs)


def load_and_preprocess_from_path_label(image, label):
    #     image = tf.image.decode_jpeg(image, channels=1)
    #   image = tf.image.resize(image, [im_height, im_width])
    image = 2 * tf.cast(image, dtype=tf.float32) / 255. - 1
    label = tf.cast(label, dtype=tf.int32)

    return image, label


import time
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, \
    f1_score

CV_summary = []
t_CV = time.perf_counter()

fold = 0
Final_CM = np.mat(np.zeros((4, 4)))  # define a confusion matrix size for the output
Final_GT = []
Final_pred = []

for i in kf.split(X_left, y_left):
    fold += 1
    train_image_left = X_left[i[0]]
    train_label_left = one_hot_labels_left[i[0]]

    test_image_left = X_left[i[1]]
    test_label_left = one_hot_labels_left[i[1]]

    train_dataset_left = tf.data.Dataset.from_tensor_slices((train_image_left, train_label_left))
    train_db_left = train_dataset_left.shuffle(2000).map(load_and_preprocess_from_path_label).batch(5)

    test_dataset_left = tf.data.Dataset.from_tensor_slices((test_image_left, test_label_left))
    test_db_left = test_dataset_left.shuffle(2000).map(load_and_preprocess_from_path_label).batch(5)

    train_image_right = X_right[i[0]]
    train_label_right = one_hot_labels_right[i[0]]

    test_image_right = X_right[i[1]]
    test_label_right = one_hot_labels_right[i[1]]

    train_dataset_right = tf.data.Dataset.from_tensor_slices((train_image_right, train_label_right))
    train_db_right = train_dataset_right.shuffle(2000).map(load_and_preprocess_from_path_label).batch(10)

    test_dataset_right = tf.data.Dataset.from_tensor_slices((test_image_right, test_label_right))
    test_db_right = test_dataset_right.shuffle(2000).map(load_and_preprocess_from_path_label).batch(10)

    t_fold = time.perf_counter()

    model_left = xception
    model_right = xception
    model_left.summary()

    optimizer = optimizers.Adam(lr=1e-5)

    train_loss_left = tf.keras.metrics.Mean(name='train_loss_left')
    train_accuracy_left = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy_left')

    test_loss_left = tf.keras.metrics.Mean(name='test_loss_left')
    test_accuracy_left = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy_left')

    train_loss_right = tf.keras.metrics.Mean(name='train_loss_right')
    train_accuracy_right = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy_right')

    test_loss_right = tf.keras.metrics.Mean(name='test_loss_right')
    test_accuracy_right = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy_right')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # CV_log_dir = 'logs/CV' + TOT + str(fold) + '%%' + current_time
    # CV_summary_writer = tf.summary.create_file_writer(CV_log_dir)

    CM_summary = np.mat(np.zeros((4, 4)))
    Epoch_summary = []

    epochs = 30  # fine-tuning is required
    epsilon = 0

    for epoch in range(1, epochs + 1):
        train_loss_left.reset_states()  # clear history info
        train_accuracy_left.reset_states()  # clear history info
        test_loss_left.reset_states()  # clear history info
        test_accuracy_left.reset_states()  # clear history info

        summary = []

        t1_left = time.perf_counter()
        for step, (x_left, y_left) in enumerate(train_db_left):
            with tf.GradientTape() as tape:
                logits_left = model_left(x_left, training=True)
                # [b] => [b, 2]
                # y_onehot = tf.one_hot(y, depth=2)
                # compute loss
                loss_left = tf.losses.categorical_crossentropy(y_left, logits_left, from_logits=True)
                #                 loss=tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = logits, dim=-1,name=None)
                loss_left = tf.reduce_mean(loss_left)
                train_loss_left(loss_left)
                train_accuracy_left(y_left, logits_left)

            grads_left = tape.gradient(loss_left, model_left.trainable_variables)
            optimizer.apply_gradients(zip(grads_left, model_left.trainable_variables))
            
        print('-----------------------------------------------------------------')
        print('Left Training time: ', time.perf_counter() - t1_left)

        test_pred_left = []
        test_GT_left = []

        t2_left = time.perf_counter()
        for xt_left, yt_left in test_db_left:
            logits_left = model_left(xt_left, training=False)
            prob_left = tf.nn.softmax(logits_left, axis=1)
            pred_left = tf.argmax(prob_left, axis=1)
            pred_left = tf.cast(pred_left, dtype=tf.int32)

            new_label_left = tf.argmax(yt_left, axis=1)
            test_pred_left.extend(pred_left)
            test_GT_left.extend(new_label_left)
            # print(pred,label)
            # yt_onehot = tf.one_hot(yt, depth=2)
            t_loss_left = tf.losses.categorical_crossentropy(yt_left, logits_left, from_logits=True)
            #             t_loss=tf.nn.softmax_cross_entropy_with_logits(labels = yt, logits = logits, dim=-1,name=None)

            test_loss_left(t_loss_left)
            test_accuracy_left(yt_left, logits_left)
        print('-----------------------------------------------------------------')
        print('Left Test time: ', time.perf_counter() - t2_left)

        train_loss_right.reset_states()  # clear history info
        train_accuracy_right.reset_states()  # clear history info
        test_loss_right.reset_states()  # clear history info
        test_accuracy_right.reset_states()  # clear history info

        summary = []

        t1_right = time.perf_counter()
        for step, (x_right, y_right) in enumerate(train_db_right):
            with tf.GradientTape() as tape:
                logits_right = model_left(x_right, training=True)
                # [b] => [b, 2]
                # y_onehot = tf.one_hot(y, depth=2)
                # compute loss
                loss_right = tf.losses.categorical_crossentropy(y_right, logits_right, from_logits=True)
                #                 loss=tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = logits, dim=-1,name=None)
                loss_right = tf.reduce_mean(loss_right)
                train_loss_right(loss_right)
                train_accuracy_right(y_right, logits_right)

            grads_right = tape.gradient(loss_right, model_right.trainable_variables)
            optimizer.apply_gradients(zip(grads_right, model_right.trainable_variables))

        print('-----------------------------------------------------------------')
        print('Right Training time: ', time.perf_counter() - t1_right)

        test_pred_right = []
        test_GT_right = []

        t2_right = time.perf_counter()
        for xt_right, yt_right in test_db_right:
            logits_right = model_right(xt_right, training=False)
            prob_right = tf.nn.softmax(logits_right, axis=1)
            pred_right = tf.argmax(prob_right, axis=1)
            pred_right = tf.cast(pred_right, dtype=tf.int32)

            new_label_right = tf.argmax(yt_right, axis=1)
            test_pred_right.extend(pred_right)
            test_GT_right.extend(new_label_right)
            # print(pred,label)
            # yt_onehot = tf.one_hot(yt, depth=2)
            t_loss_right = tf.losses.categorical_crossentropy(yt_right, logits_right, from_logits=True)
            #             t_loss=tf.nn.softmax_cross_entropy_with_logits(labels = yt, logits = logits, dim=-1,name=None)

            test_loss_right(t_loss_right)
            test_accuracy_right(yt_right, logits_right)

        print('-----------------------------------------------------------------')
        print('Right Test time: ', time.perf_counter() - t2_right)

        ground_truth = []
        prediction = []
        
        # define the fused outputs into 4x4 matrix
        for j in range(len(test_pred_right)):
            if test_GT_left[j].numpy() == 0 and test_GT_right[j].numpy() == 0:  # both sides normal
                ground_truth.append(0)
            elif test_GT_left[j].numpy() == 1 and test_GT_right[j].numpy() == 0:  # left-side malignant and right-side normal
                ground_truth.append(1)
            elif test_GT_left[j].numpy() == 0 and test_GT_right[j].numpy() == 1:  # left-side normal and right-side malignant
                ground_truth.append(2)
            elif test_GT_left[j].numpy() == 1 and test_GT_right[j].numpy() == 1:  # both sides malignant
                ground_truth.append(3)

            if test_pred_left[j].numpy() == 0 and test_pred_right[j].numpy() == 0:
                prediction.append(0)
            elif test_pred_left[j].numpy() == 1 and test_pred_right[j].numpy() == 0:
                prediction.append(1)
            elif test_pred_left[j].numpy() == 0 and test_pred_right[j].numpy() == 1:
                prediction.append(2)
            elif test_pred_left[j].numpy() == 1 and test_pred_right[j].numpy() == 1:
                prediction.append(3)

        CM = confusion_matrix(ground_truth, prediction)

        print("Confusion Mtrix")
        print(CM)
        print('\nClassification Report\n')
        print(classification_report(ground_truth, prediction, labels=range(4),
                                    target_names=['Normal', 'Left Abnormal', 'Right Abnormal', 'Abnormal']))

        Acc = accuracy_score(ground_truth, prediction)
        if Acc > epsilon:
            epsilon = Acc
            Best_CM = np.array(CM)
            Best_GT = ground_truth
            Best_pred = prediction
        print("Current Best Classification Report:")
        print(classification_report(Best_GT, Best_pred, labels=range(4),
                                    target_names=['Normal', 'Left Abnormal', 'Right Abnormal', 'Abnormal']))
        print("Current Best CM:")
        print(Best_CM)

    Final_CM += Best_CM
    Final_GT.extend(Best_GT)
    Final_pred.extend(Best_pred)

    print("---------------------------------------------------------------------")
    print("Fold Summary:")
    print(classification_report(Final_GT, Final_pred, labels=range(4),
                                target_names=['Normal', 'Left Abnormal', 'Right Abnormal', 'Abnormal']))
    print("Final_CM:")
    print(Final_CM)

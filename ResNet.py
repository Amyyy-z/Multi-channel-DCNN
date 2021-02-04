# @Date                 : 2021-02-04
# @Author               : Xinyu Zhang (Amy)
# @Python               : 3.7
# @Tensorflow Version   : 2.1.0

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

with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([10000, 1000])
    cpu_b = tf.random.normal([1000, 2000])
    print(cpu_a.device, cpu_b.device)
with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([10000, 1000])
    gpu_b = tf.random.normal([1000, 2000])
    print(gpu_a.device, gpu_b.device)
def cpu_run():
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a, cpu_b)
    return c
def gpu_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(gpu_a, gpu_b)
    return c
# warm up
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('warmup:', cpu_time, gpu_time)

cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('run time:', cpu_time, gpu_time)


from PIL import Image
import glob
import cv2
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# import all the images in the file Left & Right
image_list = []
files = glob.glob(r"C:\...\Left\*.png")

for filename in files:
    print(filename)
    image = cv2.imread(filename)
    # image = tf.image.resize(image, [125, 125]) #resize images into required height and width
    image_list.append(image)  # convert images into array
    # print(filename)
print('image_list shape:', np.array(image_list).shape)  # check whether the right amount of images were imported

# convert image list into array
image_list = np.asarray(image_list)

image_list
print(type(image_list))

import matplotlib.pyplot as plt
% matplotlib
inline
import numpy as np

X = image_list.reshape(1224, 3, 224, 224).transpose(0, 2, 3, 1).astype("uint8")  # reshape images, 1224 is the number of input images
# 224*224*3 pixels

plt.imshow(image_list[1018])

# Labels
df = pd.read_csv("Left.csv")
df.info()

# Extract type from table
Label = df["Type"]

# Stack into array
label = np.asarray(Label)
label


def one_hot_encode(vec, vals=2):
    # to one-hot encode the 4- possible labesl
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out


# set up images and labels
class CifarHelper():
    def __init__(self):
        self.i = 0

        # self.all_train_batches = [X]

        self.images = None
        self.labels = None

    def set_up_images(self):
        print("Setting up images and labels")
        self.images = np.vstack([X])
        all_len = len(self.images)

        self.images = self.images.reshape(all_len, 3, 224, 224).transpose(0, 2, 3, 1) / 255
        self.labels = one_hot_encode(np.hstack([label]), 2)


# before tensorflow run:
ch = CifarHelper()
ch.set_up_images()

# Check the image and its label
index = 1090
plt.imshow(X[index])
print(label[index])


# Encoding data
def vectorize_sequences(sequences, dimension=1000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# Encode labels
def to_one_hot(labels, dimension=2):  # number can be updated depend on the output class
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


one_hot_labels = to_one_hot(label)

import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold

# k-fold stratified CV for maintaining the imbalanced dataset ratio for both training and testing sets
kf = StratifiedKFold(n_splits=10)
kf.get_n_splits(X, label)

# 10-fold stratified CV split
for train_index, test_index in kf.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

from tensorflow.keras import layers, Model, Sequential, regularizers


# Model buid
class BasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                   padding="SAME", use_bias=False, kernel_regularizer=regularizers.l2(0.001)
                                   )
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                   padding="SAME", use_bias=False, kernel_regularizer=regularizers.l2(0.001)
                                   )
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.downsample = downsample
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([identity, x])
        x = self.relu(x)

        return x


class Bottleneck(layers.Layer):
    expansion = 4

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False, name="conv1",
                                   kernel_regularizer=regularizers.l2(0.001)
                                   )
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False,
                                   strides=strides, padding="SAME", name="conv2",
                                   kernel_regularizer=regularizers.l2(0.001)
                                   )
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm")
        # -----------------------------------------
        self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False, name="conv3",
                                   kernel_regularizer=regularizers.l2(0.001)
                                   )
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm")
        # -----------------------------------------
        self.relu = layers.ReLU()
        self.downsample = downsample
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([x, identity])
        x = self.relu(x)

        return x


def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    downsample = None
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                          use_bias=False, name="conv1", kernel_regularizer=regularizers.l2(0.001)
                          ),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")

    layers_list = []
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))

    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))

    return Sequential(layers_list, name=name)


def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=2, include_top=True):
    # (None, 224, 224, 3)
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="SAME", use_bias=False, name="conv1", kernel_regularizer=regularizers.l2(0.001)
                      )(input_image)

    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4")(x)

    x = Dropout(0.5)(x)  # dropout layer can be updated depends on whether overfitting is shown

    if include_top:
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        predict = x

    model = Model(inputs=input_image, outputs=predict)

    return model


def resnet6(im_width=224, im_height=224, num_classes=1000):
    return _resnet(BasicBlock, [1, 1, 1, 1], im_width, im_height, num_classes)


def resnet18(im_width=224, im_height=224, num_classes=1000):
    return _resnet(BasicBlock, [2, 2, 2, 2], im_width, im_height, num_classes)


def resnet34(im_width=224, im_height=224, num_classes=1000):
    return _resnet(BasicBlock, [3, 4, 6, 3], im_width, im_height, num_classes)


def resnet50(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(Bottleneck, [3, 4, 6, 3], im_width, im_height, num_classes, include_top)


def resnet101(im_width=224, im_height=224, num_classes=1000, include_top=True):
    return _resnet(Bottleneck, [3, 4, 23, 3], im_width, im_height, num_classes, include_top)


def load_and_preprocess_from_path_label(image, label):
    image = 2 * tf.cast(image, dtype=tf.float32) / 255. - 1
    label = tf.cast(label, dtype=tf.int32)

    return image, label


import time
from tensorflow.keras import optimizers

# Train and test model through CV
CV_summary = []
CM_summary_final = np.mat(np.zeros((2, 2)))  # output a 2*2 matrix for confusion matrix
t_CV = time.perf_counter()

fold = 0

for i in kf.split(X, y):
    fold += 1
    train_image = X[i[0]]
    train_label = one_hot_labels[i[0]]

    test_image = X[i[1]]
    test_label = one_hot_labels[i[1]]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_image, train_label))
    train_db = train_dataset.shuffle(1000).map(load_and_preprocess_from_path_label).batch(15)  # set up batch size 15

    test_dataset = tf.data.Dataset.from_tensor_slices((test_image, test_label))
    test_db = test_dataset.shuffle(1000).map(load_and_preprocess_from_path_label).batch(15)

    # print(train_db, test_db)

    # print(train_image[1])

    t_fold = time.perf_counter()

    model = resnet6(num_classes=2)
    model.summary()

    optimizer = optimizers.Adam(lr=1e-5)  # select optimizer and learning rate

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # CV_log_dir = 'logs/CV' + TOT + str(fold) + '%%' + current_time
    # CV_summary_writer = tf.summary.create_file_writer(CV_log_dir)

    CM_summary = np.mat(np.zeros((2, 2)))
    Epoch_summary = []

    epochs = 30  # number of epochs can be updated based on the model's convergence

    for epoch in range(1, epochs + 1):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        test_loss.reset_states()  # clear history info
        test_accuracy.reset_states()  # clear history info
        summary = []

        t1 = time.perf_counter()
        for step, (x, y) in enumerate(train_db):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
                loss = tf.reduce_mean(loss)
                train_loss(loss)
                train_accuracy(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
        print('-----------------------------------------------------------------')
        print('Training time: ', time.perf_counter() - t1)

        test_pred = []
        test_GT = []

        t2 = time.perf_counter()
        for xt, yt in test_db:
            logits = model(xt, training=False)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            new_label = tf.argmax(yt, axis=1)
            test_pred.extend(pred)
            test_GT.extend(new_label)
            t_loss = tf.losses.categorical_crossentropy(yt, logits, from_logits=True)
            test_loss(t_loss)
            test_accuracy(yt, logits)

        CM = tf.math.confusion_matrix(test_GT, test_pred)
        TP = CM[1, 1]
        TN = CM[0, 0]
        FP = CM[0, 1]
        FN = CM[1, 0]

        Acc = ((TP + TN) / (TP + TN + FP + FN))
        PPV = TP / (TP + FP)
        Sensitivity = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        F1 = 2 * (PPV * Sensitivity) / (PPV + Sensitivity)
        NPV = TN / (TN + FN)

        if epoch > 20:  # 25
            summary = [train_loss.result().numpy(), train_accuracy.result().numpy(), test_loss.result().numpy(),
                       Acc.numpy(), F1, Sensitivity, Specificity, PPV, NPV]
            Epoch_summary.append(summary)

        print('Test time: ', time.perf_counter() - t2)
        template1 = 'Fold {}, Epoch {}'
        template2 = 'Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template1.format(fold, epoch))
        print(template2.format(loss,
                               train_accuracy.result(),
                               test_loss.result(),
                               Acc))

        print(CM)
        print('Acc:', float(Acc), 'F1:', float(F1), 'Sensitivity:', float(Sensitivity), 'Specificity:',
              float(Specificity),
              'PPV:', float(PPV), 'NPV:', float(NPV))
        print('-----------------------------------------------------------------')

    print('Fold time: ', time.perf_counter() - t_fold)
    print('Summary for fold: ', fold)
    print(Epoch_summary)
    epoch_mean = np.mean(Epoch_summary, axis=0)
    print('Mean:')
    print(epoch_mean)

    CV_summary.append(epoch_mean)

print('_________________________________________________________________')
print('Cross validation summary: ')
print('Total time: ', time.perf_counter() - t_CV)
print(CV_summary)
print('Mean:')
print(np.mean(CV_summary, axis=0))

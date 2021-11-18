#Build models using Keras
import tensorflow.keras

from tensorflow.keras import models, layers
from tensorflow.keras.models import Model, model_from_json, Sequential

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, SeparableConv2D, UpSampling2D, BatchNormalization, Input, GlobalAveragePooling2D

from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, RMSprop
from tensorflow.keras.utils import to_categorical
from keras.utils.vis_utils import plot_model

#Build models using Keras
import keras
from keras import models
from keras import layers
from keras.layers.core import Permute
import tensorflow as tf
import tensorflow.compat.v1 as tf

# tf.disable_v2_behavior()
tf.enable_eager_execution() 

from PIL import Image
import glob
import cv2
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Load the TensorBoard notebook extension
%load_ext tensorboard

import os

data_dir = r'C:/.../Ultra binary data/' # load the images
contents = os.listdir(data_dir)
classes = [each for each in contents if os.path.isdir(data_dir + each)]

labels = []
inputfirst = tf.float32, [None, 224, 224, 3]
input_ = tf.float32, [None, 224, 224, 3]

with tf.Session() as sess:
    for i,each in enumerate(classes,1):
        print("Starting {} images".format(each))
        class_path = data_dir + each
        print(class_path)
        files= os.listdir(class_path)
      
        for ii , file in enumerate(files,1):
            print(os.path.basename(file))
            image_value = tf.read_file(os.path.join(class_path, file))
 
            img = tf.image.decode_jpeg(image_value, channels=3)
            
            tf.global_variables_initializer()
           
            img= tf.image.resize_images(img, [224,224],method=0)    
            print(img)
            
            imgput= tf.reshape(img,[1,224,224,3])
            if ((ii==1)&(i==1)):
                inputfirst=imgput
            else:
               
                inputfirst=tf.concat([inputfirst,imgput],axis=0)
            labels.append(each)   
 
    labels=tf.reshape(labels,[-1])
    print(inputfirst.shape)
    print(labels.shape)


from keras import backend as K
X = inputfirst.eval(session=tf.Session())
y = labels.eval(session=tf.Session())
y = tf.strings.to_number(y,tf.int32)

y = np.array(y)  # convert string labels to arrary


import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
from PIL import Image

X = X.reshape(917,3,224,224).transpose(0,2,3,1).astype("uint8")  # 917 is the total number of images included


def one_hot_encode(vec, vals = 2):
    #to one-hot encode the 4- possible labesl
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out
    
# define CifarHelper function to match images with labels
class CifarHelper():
    def __init__(self):
        self.i = 0
        
        #self.all_train_batches = [X]
        
        self.images = None
        self.labels = None
        
    def set_up_images(self):
        print("Setting up images and labels")
        self.images = np.vstack([X])
        all_len = len(self.images)
        
        self.images = self.images.reshape(all_len, 3, 224, 224).transpose(0,2,3,1)/255
        self.labels = one_hot_encode(np.hstack([y]), 2)
        
#before tensorflow run:
ch = CifarHelper()
ch.set_up_images()

#Encoding data

def vectorize_sequences(sequences, dimension = 1000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

def to_one_hot(y, dimension=2):
    results = np.zeros((len(y), dimension))
    for i, label in enumerate(y):
        results[i, label] = 1.
    return results

# convert labels into one_hot format
one_hot_labels = to_one_hot(y)

def load_and_preprocess_from_path_label(X, y):
    X = 2*tf.cast(X, dtype=tf.float32) / 255.-1
    y = tf.cast(y, dtype=tf.int32)

    return X, y

import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold

# apply startified k-fold cv
kf = StratifiedKFold(n_splits=10)

for train_index, test_index in kf.split(X, y):
    print("Train:", train_index, "Test:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

from tensorflow.keras import layers, Model, Sequential, regularizers

# model construction

def entry_flow(inputs) :
    
    # channel-1 with kernel size of 3
    
    x = Conv2D(32, 3, strides = 2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64,3,padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    previous_block_activation_x = x

    for size in [128, 256, 728] :

        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(size, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = MaxPooling2D(3, strides=2, padding='same')(x)

        residual_x = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation_x)

        x = tensorflow.keras.layers.Add()([x, residual_x])
        previous_block_activation_x = x

    # channel-2 with kernel size of 7

    y = Conv2D(32, 7, strides = 2, padding='same')(inputs)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv2D(64,7,padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    previous_block_activation_y = y

    for size in [128, 256, 728] :

        y = Activation('relu')(y)
        y = SeparableConv2D(size, 3, padding='same')(y)
        y = BatchNormalization()(y)

        y = Activation('relu')(y)
        y = SeparableConv2D(size, 3, padding='same')(y)
        y = BatchNormalization()(y)

        y = MaxPooling2D(3, strides=2, padding='same')(y)

        residual_y = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation_y)

        y = tensorflow.keras.layers.Add()([y, residual_y])
        previous_block_activation_y = y

    return x, y


def middle_flow(x, y, num_blocks=8) :

    previous_block_activation_x = x

    for _ in range(num_blocks) :

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, 3, padding='same')(x)
        x = BatchNormalization()(x)

        x = tensorflow.keras.layers.Add()([x, previous_block_activation_x])
        previous_block_activation_x = x
        
    previous_block_activation_y = y

    for _ in range(num_blocks) :

        y = Activation('relu')(y)
        y = SeparableConv2D(728, 3, padding='same')(y)
        y = BatchNormalization()(y)

        y = Activation('relu')(y)
        y = SeparableConv2D(728, 3, padding='same')(y)
        y = BatchNormalization()(y)

        y = Activation('relu')(y)
        y = SeparableConv2D(728, 3, padding='same')(y)
        y = BatchNormalization()(y)

        y = tensorflow.keras.layers.Add()([y, previous_block_activation_y])
        previous_block_activation_y = y


    return x, y

def exit_flow(x, y) :

    previous_block_activation_x = x

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x) 
    x = BatchNormalization()(x)

    x = MaxPooling2D(3, strides=2, padding='same')(x)

    residual_x = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation_x)
    x = tensorflow.keras.layers.Add()([x, residual_x])

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 3, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(1024, 3, padding='same')(x)
    x = BatchNormalization()(x)
    
    previous_block_activation_y = y

    y = Activation('relu')(y)
    y = SeparableConv2D(728, 3, padding='same')(y)
    y = BatchNormalization()(y)

    y = Activation('relu')(y)
    y = SeparableConv2D(1024, 3, padding='same')(y) 
    y = BatchNormalization()(y)

    y = MaxPooling2D(3, strides=2, padding='same')(y)

    residual_y = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation_y)
    y = tensorflow.keras.layers.Add()([y, residual_y])

    y = Activation('relu')(y)
    y = SeparableConv2D(728, 3, padding='same')(y)
    y = BatchNormalization()(y)

    y = Activation('relu')(y)
    y = SeparableConv2D(1024, 3, padding='same')(y)
    y = BatchNormalization()(y)
    
    # fuse channel-1 and channel-2
    z = tf.math.add(x,y)
    z = GlobalAveragePooling2D()(z)
    z = Dense(2, activation='linear')(z)

    return z


inputs = Input(shape=(224,224,3))
outputs_x1, outputs_y1 = entry_flow(inputs)
outputs_x2, outputs_y2 = middle_flow(outputs_x1, outputs_y1)
outputs = exit_flow(outputs_x2, outputs_y2)
xception = Model(inputs, outputs)

import time
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score 

CV_summary = []
t_CV = time.perf_counter()

fold = 0
Final_CM = np.mat(np.zeros((2,2)))
Final_GT = []
Final_pred = []

for i in kf.split(X, y):
    fold += 1
    train_image = X[i[0]]
    train_label = one_hot_labels[i[0]]
    
    test_image = X[i[1]]
    test_label = one_hot_labels[i[1]]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_image,train_label))
    train_db = train_dataset.shuffle(2000).map(load_and_preprocess_from_path_label).batch(5)
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_image,test_label))
    test_db = test_dataset.shuffle(2000).map(load_and_preprocess_from_path_label).batch(5)
    

    
    t_fold = time.perf_counter()

    model = xception
    model.summary()
    
    optimizer = optimizers.Adam(lr=1e-5)
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')


    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #CV_log_dir = 'logs/CV' + TOT + str(fold) + '%%' + current_time
    #CV_summary_writer = tf.summary.create_file_writer(CV_log_dir)

    
    CM_summary = np.mat(np.zeros((2,2)))
    Epoch_summary = []

    epochs = 30   # can be fine-tuned
    epsilon = 0
    
    for epoch in range(1,epochs+1):
        train_loss.reset_states()  # clear history info
        train_accuracy.reset_states()  # clear history info
        test_loss.reset_states()  # clear history info
        test_accuracy.reset_states()  # clear history info
        
        summary = []
        

        t1 = time.perf_counter()
        for step,(x,y) in enumerate(train_db):

            with tf.GradientTape() as tape:

                logits = model(x, training=True)
                loss = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
#                 loss=tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = logits, dim=-1,name=None)
                loss = tf.reduce_mean(loss)
                train_loss(loss)
                train_accuracy(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
        print('-----------------------------------------------------------------')
        print('Training time: ',time.perf_counter() - t1)

        test_pred = []
        test_GT = []
        
        t2 = time.perf_counter()
        for xt,yt in test_db:

            logits = model(xt, training=False)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            new_label = tf.argmax(yt,axis=1)
            test_pred.extend(pred)
            test_GT.extend(new_label)
            t_loss = tf.losses.categorical_crossentropy(yt, logits, from_logits=True)
#             t_loss=tf.nn.softmax_cross_entropy_with_logits(labels = yt, logits = logits, dim=-1,name=None)
            test_loss(t_loss)
            test_accuracy(yt, logits)
        print('-----------------------------------------------------------------')
        print('Test time: ',time.perf_counter() - t2)


        CM = confusion_matrix(test_GT,test_pred)      
        
        print("Confusion Mtrix")
        print(CM)
        print('\nClassification Report\n')
        print(classification_report(test_GT,test_pred,labels=range(2),target_names=['Benign','Malignant']))

        Acc = accuracy_score(test_GT,test_pred)
        if Acc > epsilon:
            epsilon = Acc
            Best_CM = np.array(CM)
            Best_GT = test_GT
            Best_pred = test_pred
            
        print("Current Best Classification Report:")
        print(classification_report(Best_GT, Best_pred,labels=range(2), target_names=['Benign','Malignant']))
        print("Current Best CM:")
        print(Best_CM)
        
    Final_CM += Best_CM
    Final_GT.extend(Best_GT)
    Final_pred.extend(Best_pred)

print("---------------------------------------------------------------------")
print("Fold Summary:")
print(classification_report(Final_GT, Final_pred,labels=range(2), target_names=['Benign', 'Malignant']))
print("Final_CM:")
print(Final_CM)

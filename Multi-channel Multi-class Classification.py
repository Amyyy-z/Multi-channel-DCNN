#import libraries

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
#tf.disable_v2_behavior()
tf.enable_eager_execution() 

from PIL import Image
import glob
import cv2
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

#import all the images in the file Left & Right for CT scans, or all images from ultrasound
image_list_left = []
files = glob.glob (r"C:\Users\admin\Desktop\WCCI - v2\Left\*.png")

for filename in files:
  print(filename)
  image = cv2.imread(filename)
  image_list_left.append(image) #convert images into array

#apply the same for the right side images
image_list_right = []
files = glob.glob (r"C:\Users\admin\Desktop\WCCI - v2\Right\*.png")

for filename in files:
  print(filename)
  image = cv2.imread(filename)
  image_list_right.append(image) #convert images into array

#print the shape of the image sets
print('image_list shape:', np.array(image_list_left).shape)
print('image_list shape:', np.array(image_list_right).shape)

#convert list into array
image_list_left = np.asarray(image_list_left)
image_list_right = np.asarray(image_list_right)

#reshape the images
X_left = image_list_left.reshape(977,3,224,224).transpose(0,2,3,1).astype("uint8") #this is adjusted based on the input volumes
X_right = image_list_right.reshape(977,3,224,224).transpose(0,2,3,1).astype("uint8")

#import labels for the images
df_left = pd.read_csv(r"C:\Users\admin\Desktop\WCCI - v2\Left_v2.csv")
df_right = pd.read_csv(r"C:\Users\admin\Desktop\WCCI - v2\Right_v2.csv")

#Extract labels from df table
y_left = df_left["Class"]
y_right = df_right["Class"]

#Stack labels into array
y_left = np.asarray(y_left)
y_right = np.asarray(y_right)

#Encode the labels for both left and right image sets
def one_hot_encode(vec, vals = 6):
  #to one-hot encode the 6- possible labesl
  n = len(vec)
  out = np.zeros((n, vals))
  out[range(n), vec] = 1
  return out

#Map the image with labels for both side
class CifarHelper():
  def __init__(self):
    self.i = 0
    
    self.images = None
    self.labels = None
        
  def set_up_images(self):
    print("Setting up images and labels")
    self.images = np.vstack([X_left])
    all_len = len(self.images)
        
    self.images = self.images.reshape(all_len,3, 224,224).transpose(0,2,3,1)/255
    self.labels = one_hot_encode(np.hstack([y_left]), 6)
    
#before tensorflow run:
ch = CifarHelper()
ch.set_up_images()

#repeat this step for the right side images
class CifarHelper():
  def __init__(self):
    self.i = 0
        
    self.images = None
    self.labels = None
        
  def set_up_images(self):
    print("Setting up images and labels")
    self.images = np.vstack([X_right])
    all_len = len(self.images)
        
    self.images = self.images.reshape(all_len,3, 224,224).transpose(0,2,3,1)/255
    self.labels = one_hot_encode(np.hstack([y_right]), 6)
        
ch = CifarHelper()
ch.set_up_images()
      
#Encode the labels
def vectorize_sequences(sequences, dimension = 1000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.
  return results
  
def to_one_hot(y_left, dimension=6):
  results = np.zeros((len(y_left), dimension))
  for i, label in enumerate(y_left):
    results[i, label] = 1.
  return results

#apply again for the right-side labels
def to_one_hot(y_right, dimension=6):
  results = np.zeros((len(y_right), dimension))
  for i, label in enumerate(y_right):
    results[i, label] = 1.
  return results

one_hot_labels_left = to_one_hot(y_left)
one_hot_labels_right = to_one_hot(y_right)

#image setup
def load_and_preprocess_from_path_label(X_left, y_left):
  X_left = 2*tf.cast(X_left, dtype=tf.float32) / 255.-1
  y_left = tf.cast(y_left, dtype=tf.int32)
  return X_left, y_left

def load_and_preprocess_from_path_label(X_right, y_right):
  X_right = 2*tf.cast(X_right, dtype=tf.float32) / 255.-1
  y_right = tf.cast(y_right, dtype=tf.int32)
  return X_right, y_right

import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import StratifiedKFold

#training and testing splits with stratified CV
kf = StratifiedKFold(n_splits=10, random_state=1)

kf.get_n_splits(X_left, y_left)
kf.get_n_splits(X_right, y_right)

for train_index, test_index in kf.split(X_left, y_left):
  X_train_left, X_test_left = X_left[train_index], X_left[test_index]
  X_train_right, X_test_right = X_right[train_index], X_right[test_index]

  y_train_left, y_test_left = y_left[train_index], y_left[test_index]
  y_train_right, y_test_right = y_right[train_index], y_right[test_index]

#model build
#entry flow
def entry_flow(inputs) :
  x = Conv2D(32, 1, strides = 2, padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  x = Conv2D(64,1,padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)

  previous_block_activation_x = x

  for size in [128, 256, 728] :
    x = Activation('relu')(x)
    x = SeparableConv2D(size, 1, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(size, 1, padding='same')(x)
    x = BatchNormalization()(x)

    x = MaxPooling2D(3, strides=2, padding='same')(x)

    residual_x = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation_x)

    x = tf.keras.layers.Add()([x, residual_x])
    previous_block_activation_x = x

  y = Conv2D(32, 7, strides = 2, padding='same')(inputs)
  y = BatchNormalization()(y)
  y = Activation('relu')(y)

  y = Conv2D(64,7,padding='same')(y)
  y = BatchNormalization()(y)
  y = Activation('relu')(y)

  previous_block_activation_y = y

  for size in [128, 256, 728] :
    y = Activation('relu')(y)
    y = SeparableConv2D(size, 7, padding='same')(y)
    y = BatchNormalization()(y)

    y = Activation('relu')(y)
    y = SeparableConv2D(size, 7, padding='same')(y)
    y = BatchNormalization()(y)

    y = MaxPooling2D(3, strides=2, padding='same')(y)

    residual_y = Conv2D(size, 1, strides=2, padding='same')(previous_block_activation_y)

    y = tf.keras.layers.Add()([y, residual_y])
    previous_block_activation_y = y

  return x, y

#middle flow
def middle_flow(x, y, num_blocks=8) :
  previous_block_activation_x = x
  for _ in range(num_blocks) :
    x = Activation('relu')(x)
    x = SeparableConv2D(728, 1, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 1, padding='same')(x)
    x = BatchNormalization()(x)

    x = Activation('relu')(x)
    x = SeparableConv2D(728, 1, padding='same')(x)
    x = BatchNormalization()(x)

    x = tf.keras.layers.Add()([x, previous_block_activation_x])
    previous_block_activation_x = x
    
  previous_block_activation_y = y
  for _ in range(num_blocks) :
    y = Activation('relu')(y)
    y = SeparableConv2D(728, 7, padding='same')(y)
    y = BatchNormalization()(y)

    y = Activation('relu')(y)
    y = SeparableConv2D(728, 7, padding='same')(y)
    y = BatchNormalization()(y)

    y = Activation('relu')(y)
    y = SeparableConv2D(728, 7, padding='same')(y)
    y = BatchNormalization()(y)

    y = tf.keras.layers.Add()([y, previous_block_activation_y])
    previous_block_activation_y = y
  return x, y

#exit flow
def exit_flow(x, y) :
  previous_block_activation_x = x
  x = Activation('relu')(x)
  x = SeparableConv2D(728, 1, padding='same')(x)
  x = BatchNormalization()(x)

  x = Activation('relu')(x)
  x = SeparableConv2D(1024, 1, padding='same')(x) 
  x = BatchNormalization()(x)

  x = MaxPooling2D(3, strides=2, padding='same')(x)

  residual_x = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation_x)
  x = tf.keras.layers.Add()([x, residual_x])

  x = Activation('relu')(x)
  x = SeparableConv2D(728, 1, padding='same')(x)
  x = BatchNormalization()(x)

  x = Activation('relu')(x)
  x = SeparableConv2D(1024, 1, padding='same')(x)
  x = BatchNormalization()(x)

  previous_block_activation_y = y
  y = Activation('relu')(y)
  y = SeparableConv2D(728, 7, padding='same')(y)
  y = BatchNormalization()(y)

  y = Activation('relu')(y)
  y = SeparableConv2D(1024, 7, padding='same')(y) 
  y = BatchNormalization()(y)

  y = MaxPooling2D(3, strides=2, padding='same')(y)

  residual_y = Conv2D(1024, 1, strides=2, padding='same')(previous_block_activation_y)
  y = tf.keras.layers.Add()([y, residual_y])

  y = Activation('relu')(y)
  y = SeparableConv2D(728, 7, padding='same')(y)
  y = BatchNormalization()(y)

  y = Activation('relu')(y)
  y = SeparableConv2D(1024, 7, padding='same')(y)
  y = BatchNormalization()(y)
     
  z = tf.math.add(x,y)
  z = GlobalAveragePooling2D()(z)
  z = Dense(6, activation='linear')(z)

  return z

inputs = Input(shape=(224,224,3))
outputs_x1, outputs_y1 = entry_flow(inputs)
outputs_x2, outputs_y2 = middle_flow(outputs_x1, outputs_y1)
outputs = exit_flow(outputs_x2, outputs_y2)
xception = Model(inputs, outputs)

import time
from tensorflow.keras import optimizers
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score 

#training and testing starts
CV_summary = []
t_CV = time.perf_counter()

fold = 0
Final_CM = np.mat(np.zeros((16,16))) #define 16 by 16 matrix 
Final_GT = []
Final_pred = []

for i in kf.split(X_left, y_left):
    fold += 1
    train_image_left = X_left[i[0]]
    train_label_left = one_hot_labels_left[i[0]]
    
    test_image_left = X_left[i[1]]
    test_label_left = one_hot_labels_left[i[1]]

    train_dataset_left = tf.data.Dataset.from_tensor_slices((train_image_left,train_label_left))
    train_db_left = train_dataset_left.shuffle(2000).map(load_and_preprocess_from_path_label).batch(5)
    
    test_dataset_left = tf.data.Dataset.from_tensor_slices((test_image_left,test_label_left))
    test_db_left = test_dataset_left.shuffle(2000).map(load_and_preprocess_from_path_label).batch(5)

    train_image_right = X_right[i[0]]
    train_label_right = one_hot_labels_right[i[0]]
    
    test_image_right = X_right[i[1]]
    test_label_right = one_hot_labels_right[i[1]]

    train_dataset_right = tf.data.Dataset.from_tensor_slices((train_image_right,train_label_right))
    train_db_right = train_dataset_right.shuffle(2000).map(load_and_preprocess_from_path_label).batch(5)
    
    test_dataset_right = tf.data.Dataset.from_tensor_slices((test_image_right,test_label_right))
    test_db_right = test_dataset_right.shuffle(2000).map(load_and_preprocess_from_path_label).batch(5)
    
    t_fold = time.perf_counter()

    model_left = xception
    model_right = xception
    model_left.summary()
    
    optimizer = optimizers.Adam(lr=1e-5) #set learning rate fixed
    
    train_loss_left = tf.keras.metrics.Mean(name='train_loss_left')
    train_accuracy_left = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy_left')

    test_loss_left = tf.keras.metrics.Mean(name='test_loss_left')
    test_accuracy_left = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy_left')
    
    train_loss_right = tf.keras.metrics.Mean(name='train_loss_right')
    train_accuracy_right = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy_right')

    test_loss_right = tf.keras.metrics.Mean(name='test_loss_right')
    test_accuracy_right = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy_right')
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #CV_log_dir = 'logs/CV' + TOT + str(fold) + '%%' + current_time
    #CV_summary_writer = tf.summary.create_file_writer(CV_log_dir)
    
    CM_summary = np.mat(np.zeros((16,16)))
    Epoch_summary = []

    epochs = 30   #35
    epsilon = 0
    
    for epoch in range(1,epochs+1):
        train_loss_left.reset_states()  # clear history info
        train_accuracy_left.reset_states()  # clear history info
        test_loss_left.reset_states()  # clear history info
        test_accuracy_left.reset_states()  # clear history info
        
        summary = [] 

        t1_left = time.perf_counter()
        for step, (x_left,y_left) in enumerate(train_db_left):

            with tf.GradientTape() as tape:

                logits_left = model_left(x_left, training=True)
                # [b] => [b, 2]
                #y_onehot = tf.one_hot(y, depth=2)
                # compute loss
                loss_left = tf.losses.categorical_crossentropy(y_left, logits_left, from_logits=True)
#                 loss_left=tf.nn.softmax_cross_entropy_with_logits(labels = y_left,logits = logits_left, dim=-1,name=None)
                loss_left = tf.reduce_mean(loss_left)
                train_loss_left(loss_left)
                train_accuracy_left(y_left, logits_left)

            grads_left = tape.gradient(loss_left, model_left.trainable_variables)
            optimizer.apply_gradients(zip(grads_left, model_left.trainable_variables))

    #     with CV_summary_writer.as_default():
    #         tf.summary.scalar('train-CrossEntropy', float(loss), step=epoch)
    #         tf.summary.scalar('train-Accuracy', float(train_accuracy.result() * 100), step=epoch)
        print('-----------------------------------------------------------------')
        print('Left Training time: ',time.perf_counter() - t1_left)

        test_pred_left = []
        test_GT_left = []
        
        t2_left = time.perf_counter()
        for xt_left,yt_left in test_db_left:

            logits_left = model_left(xt_left, training=False)
            prob_left = tf.nn.softmax(logits_left, axis=1)
            pred_left = tf.argmax(prob_left, axis=1)
            pred_left = tf.cast(pred_left, dtype=tf.int32)

            new_label_left = tf.argmax(yt_left,axis=1)
            test_pred_left.extend(pred_left)
            test_GT_left.extend(new_label_left)
            #print(pred,label)
            #yt_onehot = tf.one_hot(yt, depth=2)
            t_loss_left = tf.losses.categorical_crossentropy(yt_left, logits_left, from_logits=True)
#             t_loss_left=tf.nn.softmax_cross_entropy_with_logits(labels = yt_left, logits = logits_left, dim=-1,name=None)

            test_loss_left(t_loss_left)
            test_accuracy_left(yt_left, logits_left)
        print('-----------------------------------------------------------------')
        print('Left Test time: ',time.perf_counter() - t2_left)

        train_loss_right.reset_states()  # clear history info
        train_accuracy_right.reset_states()  # clear history info
        test_loss_right.reset_states()  # clear history info
        test_accuracy_right.reset_states()  # clear history info
        
        summary = []

        t1_right = time.perf_counter()
        for step, (x_right,y_right) in enumerate(train_db_right):

            with tf.GradientTape() as tape:

                logits_right = model_left(x_right, training=True)
                # [b] => [b, 2]
                #y_onehot = tf.one_hot(y, depth=2)
                # compute loss
                loss_right = tf.losses.categorical_crossentropy(y_right, logits_right, from_logits=True)
#                 loss_right=tf.nn.softmax_cross_entropy_with_logits(labels = y_right,logits = logits_right, dim=-1,name=None)
                loss_right = tf.reduce_mean(loss_right)
                train_loss_right(loss_right)
                train_accuracy_right(y_right, logits_right)

            grads_right = tape.gradient(loss_right, model_right.trainable_variables)
            optimizer.apply_gradients(zip(grads_right, model_right.trainable_variables))

    #     with CV_summary_writer.as_default():
    #         tf.summary.scalar('train-CrossEntropy', float(loss), step=epoch)
    #         tf.summary.scalar('train-Accuracy', float(train_accuracy.result() * 100), step=epoch)
        print('-----------------------------------------------------------------')
        print('Right Training time: ',time.perf_counter() - t1_right)

        test_pred_right = []
        test_GT_right = []
        
        t2_right = time.perf_counter()
        for xt_right,yt_right in test_db_right:

            logits_right = model_right(xt_right, training=False)
            prob_right = tf.nn.softmax(logits_right, axis=1)
            pred_right = tf.argmax(prob_right, axis=1)
            pred_right = tf.cast(pred_right, dtype=tf.int32)

            new_label_right = tf.argmax(yt_right,axis=1)
            test_pred_right.extend(pred_right)
            test_GT_right.extend(new_label_right)
            #print(pred,label)
            #yt_onehot = tf.one_hot(yt, depth=2)
            t_loss_right = tf.losses.categorical_crossentropy(yt_right, logits_right, from_logits=True)
#             t_loss_right=tf.nn.softmax_cross_entropy_with_logits(labels = yt_right, logits = logits_right, dim=-1,name=None)

            test_loss_right(t_loss_right)
            test_accuracy_right(yt_right, logits_right)
     
        print('-----------------------------------------------------------------')
        print('Right Test time: ',time.perf_counter() - t2_right)
        
        
        ground_truth = []
        prediction = []
        for j in range(len(test_pred_right)):
          #define conditions (0: normal, 1: thyroiditis, 2: cystic, 3: goiter, 4: adenoma, 5: cancer)
            if test_GT_left[j].numpy() == 0 and test_GT_right[j].numpy() == 0:
                ground_truth.append(0) 
            elif test_GT_left[j].numpy() == 0 and test_GT_right[j].numpy() == 1:
                ground_truth.append(1)
            elif test_GT_left[j].numpy() == 0 and test_GT_right[j].numpy() == 2:
                ground_truth.append(2)
            elif test_GT_left[j].numpy() == 0 and test_GT_right[j].numpy() == 3:
                ground_truth.append(3)
            elif test_GT_left[j].numpy() == 0 and test_GT_right[j].numpy() == 4:
                ground_truth.append(4)
            elif test_GT_left[j].numpy() == 0 and test_GT_right[j].numpy() == 5:
                ground_truth.append(5)

            elif test_GT_left[j].numpy() == 1 and test_GT_right[j].numpy() == 0:
                ground_truth.append(1)
            elif test_GT_left[j].numpy() == 1 and test_GT_right[j].numpy() == 1:
                ground_truth.append(1)
            elif test_GT_left[j].numpy() == 1 and test_GT_right[j].numpy() == 2:
                ground_truth.append(6)
            elif test_GT_left[j].numpy() == 1 and test_GT_right[j].numpy() == 3:
                ground_truth.append(7)
            elif test_GT_left[j].numpy() == 1 and test_GT_right[j].numpy() == 4:
                ground_truth.append(8)
            elif test_GT_left[j].numpy() == 1 and test_GT_right[j].numpy() == 5:
                ground_truth.append(9)
    
            elif test_GT_left[j].numpy() == 2 and test_GT_right[j].numpy() == 0:
                ground_truth.append(2)
            elif test_GT_left[j].numpy() == 2 and test_GT_right[j].numpy() == 1:
                ground_truth.append(6)
            elif test_GT_left[j].numpy() == 2 and test_GT_right[j].numpy() == 2:
                ground_truth.append(2)
            elif test_GT_left[j].numpy() == 2 and test_GT_right[j].numpy() == 3:
                ground_truth.append(10)
            elif test_GT_left[j].numpy() == 2 and test_GT_right[j].numpy() == 4:
                ground_truth.append(11)
            elif test_GT_left[j].numpy() == 2 and test_GT_right[j].numpy() == 5:
                ground_truth.append(12)
                
            elif test_GT_left[j].numpy() == 3 and test_GT_right[j].numpy() == 0:
                ground_truth.append(3)
            elif test_GT_left[j].numpy() == 3 and test_GT_right[j].numpy() == 1:
                ground_truth.append(7)
            elif test_GT_left[j].numpy() == 3 and test_GT_right[j].numpy() == 2:
                ground_truth.append(10)
            elif test_GT_left[j].numpy() == 3 and test_GT_right[j].numpy() == 3:
                ground_truth.append(3)
            elif test_GT_left[j].numpy() == 3 and test_GT_right[j].numpy() == 4:
                ground_truth.append(13)
            elif test_GT_left[j].numpy() == 3 and test_GT_right[j].numpy() == 5:
                ground_truth.append(14)
                
            elif test_GT_left[j].numpy() == 4 and test_GT_right[j].numpy() == 0:
                ground_truth.append(4)
            elif test_GT_left[j].numpy() == 4 and test_GT_right[j].numpy() == 1:
                ground_truth.append(8)
            elif test_GT_left[j].numpy() == 4 and test_GT_right[j].numpy() == 2:
                ground_truth.append(11)
            elif test_GT_left[j].numpy() == 4 and test_GT_right[j].numpy() == 3:
                ground_truth.append(13)
            elif test_GT_left[j].numpy() == 4 and test_GT_right[j].numpy() == 4:
                ground_truth.append(4)
            elif test_GT_left[j].numpy() == 4 and test_GT_right[j].numpy() == 5:
                ground_truth.append(15)
                
            elif test_GT_left[j].numpy() == 5 and test_GT_right[j].numpy() == 0:
                ground_truth.append(5)
            elif test_GT_left[j].numpy() == 5 and test_GT_right[j].numpy() == 1:
                ground_truth.append(9)
            elif test_GT_left[j].numpy() == 5 and test_GT_right[j].numpy() == 2:
                ground_truth.append(12)
            elif test_GT_left[j].numpy() == 5 and test_GT_right[j].numpy() == 3:
                ground_truth.append(14)
            elif test_GT_left[j].numpy() == 5 and test_GT_right[j].numpy() == 4:
                ground_truth.append(15)
            elif test_GT_left[j].numpy() == 5 and test_GT_right[j].numpy() == 5:
                ground_truth.append(5)
                

            if test_pred_left[j].numpy() == 0 and test_pred_right[j].numpy() == 0:
                prediction.append(0)
            elif test_pred_left[j].numpy() == 0 and test_pred_right[j].numpy() == 1:
                prediction.append(1)
            elif test_pred_left[j].numpy() == 0 and test_pred_right[j].numpy() == 2:
                prediction.append(2)
            elif test_pred_left[j].numpy() == 0 and test_pred_right[j].numpy() == 3:
                prediction.append(3)
            elif test_pred_left[j].numpy() == 0 and test_pred_right[j].numpy() == 4:
                prediction.append(4)
            elif test_pred_left[j].numpy() == 0 and test_pred_right[j].numpy() == 5:
                prediction.append(5)

            elif test_pred_left[j].numpy() == 1 and test_pred_right[j].numpy() == 0:
                prediction.append(1)
            elif test_pred_left[j].numpy() == 1 and test_pred_right[j].numpy() == 1:
                prediction.append(1)
            elif test_pred_left[j].numpy() == 1 and test_pred_right[j].numpy() == 2:
                prediction.append(6)
            elif test_pred_left[j].numpy() == 1 and test_pred_right[j].numpy() == 3:
                prediction.append(7)
            elif test_pred_left[j].numpy() == 1 and test_pred_right[j].numpy() == 4:
                prediction.append(8)
            elif test_pred_left[j].numpy() == 1 and test_pred_right[j].numpy() == 5:
                prediction.append(9)
                
            elif test_pred_left[j].numpy() == 2 and test_pred_right[j].numpy() == 0:
                prediction.append(2)
            elif test_pred_left[j].numpy() == 2 and test_pred_right[j].numpy() == 1:
                prediction.append(6)
            elif test_pred_left[j].numpy() == 2 and test_pred_right[j].numpy() == 2:
                prediction.append(2)
            elif test_pred_left[j].numpy() == 2 and test_pred_right[j].numpy() == 3:
                prediction.append(10)
            elif test_pred_left[j].numpy() == 2 and test_pred_right[j].numpy() == 4:
                prediction.append(11)
            elif test_pred_left[j].numpy() == 2 and test_pred_right[j].numpy() == 5:
                prediction.append(12)
                
            elif test_pred_left[j].numpy() == 3 and test_pred_right[j].numpy() == 0:
                prediction.append(3)
            elif test_pred_left[j].numpy() == 3 and test_pred_right[j].numpy() == 1:
                prediction.append(7)
            elif test_pred_left[j].numpy() == 3 and test_pred_right[j].numpy() == 2:
                prediction.append(10)
            elif test_pred_left[j].numpy() == 3 and test_pred_right[j].numpy() == 3:
                prediction.append(3)
            elif test_pred_left[j].numpy() == 3 and test_pred_right[j].numpy() == 4:
                prediction.append(13)
            elif test_pred_left[j].numpy() == 3 and test_pred_right[j].numpy() == 5:
                prediction.append(14)
                
            elif test_pred_left[j].numpy() == 4 and test_pred_right[j].numpy() == 0:
                prediction.append(4)
            elif test_pred_left[j].numpy() == 4 and test_pred_right[j].numpy() == 1:
                prediction.append(8)
            elif test_pred_left[j].numpy() == 4 and test_pred_right[j].numpy() == 2:
                prediction.append(11)
            elif test_pred_left[j].numpy() == 4 and test_pred_right[j].numpy() == 3:
                prediction.append(13)
            elif test_pred_left[j].numpy() == 4 and test_pred_right[j].numpy() == 4:
                prediction.append(4)
            elif test_pred_left[j].numpy() == 4 and test_pred_right[j].numpy() == 5:
                prediction.append(15)
                
            elif test_pred_left[j].numpy() == 5 and test_pred_right[j].numpy() == 0:
                prediction.append(5)
            elif test_pred_left[j].numpy() == 5 and test_pred_right[j].numpy() == 1:
                prediction.append(9)
            elif test_pred_left[j].numpy() == 5 and test_pred_right[j].numpy() == 2:
                prediction.append(12)
            elif test_pred_left[j].numpy() == 5 and test_pred_right[j].numpy() == 3:
                prediction.append(14)
            elif test_pred_left[j].numpy() == 5 and test_pred_right[j].numpy() == 4:
                prediction.append(15)
            elif test_pred_left[j].numpy() == 5 and test_pred_right[j].numpy() == 5:
                prediction.append(5)
                
        CM = confusion_matrix(ground_truth,prediction)      
        
       
        print('\nClassification Report\n')
        print(classification_report(ground_truth,prediction,labels=range(16), target_names=['Normal', 'Thyroiditis', 'Cystic',
                                                                          'Goiter', 'Adenoma', 'Cancer','Thy+Cys','Thy+Goi',
                                                                          'Thy+Ade','Thy+Can','Cys+Goi','Cys+Ade','Cys+Can',
                                                                          'Goi+Ade','Goi+Can','Ade+Can']))
        
        print("Confusion Matrix")
        print(CM)
        
        Acc = accuracy_score(ground_truth,prediction)
        if Acc >= epsilon: 
            epsilon = Acc
            Best_CM = np.array(CM)
            Best_GT = ground_truth
            Best_pred = prediction
        print('Current Best Classification Report')
        print(classification_report(Best_GT,Best_pred,labels=range(16), target_names=['Normal', 'Thyroiditis', 'Cystic',
                                                                          'Goiter', 'Adenoma', 'Cancer','Thy+Cys','Thy+Goi',
                                                                          'Thy+Ade','Thy+Can','Cys+Goi','Cys+Ade','Cys+Can',
                                                                          'Goi+Ade','Goi+Can','Ade+Can']))
        print("Current Best CM:")
        print(Best_CM)
        
#     Final_CM += Best_CM
    Final_GT.extend(Best_GT)
    Final_pred.extend(Best_pred)
    
#     print("GT:",len(ground_truth),type(ground_truth), ground_truth)
#     print("pred:", len(prediction), type(prediction), prediction)
#     print("F_GT:", len(Final_GT), type(Final_GT), Final_GT)
#     print("F_pre:", len(Final_pred), type(Final_pred), Final_pred)
    
    print("---------------------------------------------------------------------")
    print("Fold Summary:")
    print(classification_report(Final_GT, Final_pred,labels=range(16), target_names=['Normal', 'Thyroiditis', 'Cystic',
                                                                          'Goiter', 'Adenoma', 'Cancer','Thy+Cys','Thy+Goi',
                                                                          'Thy+Ade','Thy+Can','Cys+Goi','Cys+Ade','Cys+Can',
                                                                          'Goi+Ade','Goi+Can','Ade+Can']))
    
    Final_CM = confusion_matrix(Final_GT, Final_pred)
    
    print("Final_CM:")
    print(Final_CM)
    


# @Date                 : 2021-02-04
# @Author               : Xinyu Zhang (Amy)
# @Python               : 3.7
# @Tensorflow Version   : 2.1.0
# @Other required models can be viewed through: https://keras.io/api/applications/


import tensorflow as tf
from tensorflow.keras import layers, models, Model, Sequential, regularizers

cfgs = {
    'vgg8': [64,'M',128,'M',256,'M',512,'M',512,'M'],
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def features(cfg):
    feature_layers = []
    for v in cfg:
        if v == "M":
            feature_layers.append(layers.MaxPool2D(pool_size=[2,2], strides=2, padding='same'))
        else:
            conv2d = layers.Conv2D(v, kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
            feature_layers.append(conv2d)
    return Sequential(feature_layers, name="feature")

def VGG(feature, im_height=224, im_width=224, class_num=2):
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    x = feature(input_image)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
#     x = layers.Dense(4096, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.001))(x)
#     x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=regularizers.l2(0.001))(x)
    output = layers.Dense(class_num, activation="relu")(x)
    model = models.Model(inputs=input_image, outputs=output)
    return model

def vgg(model_name="vgg8", im_height=224, im_width=224, class_num=2):
    try:
        cfg = cfgs[model_name]
    except:
        print("Warning: model number {} not in cfgs dict!".format(model_name))
        exit(-1)
    model = VGG(features(cfg), im_height=im_height, im_width=im_width, class_num=class_num)
    return model


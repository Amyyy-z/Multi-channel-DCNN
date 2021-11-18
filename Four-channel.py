# @Date                 : 2021-02-04
# @Author               : Xinyu Zhang (Amy)
# @Python               : 3.7
# @Tensorflow Version   : 2.1.0
# @Contributor for Xception: Arjun Sarkar
# @Other required models can be viewed through: https://keras.io/api/applications/


# four-channel architecture construction

def entry_flow(inputs):
    x = Conv2D(32, 3, strides=2, padding='same')(inputs) # channel-1 with kernel size of 3
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    previous_block_activation_x = x

    for size in [128, 256, 728]:
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

    y = Conv2D(32, 7, strides=2, padding='same')(inputs)  # channel-2 with kernel size of 7
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv2D(64, 7, padding='same')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    previous_block_activation_y = y

    for size in [128, 256, 728]:
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


def middle_flow(x, y, num_blocks=8):
    previous_block_activation_x = x

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

        x = tensorflow.keras.layers.Add()([x, previous_block_activation_x])
        previous_block_activation_x = x

    previous_block_activation_y = y

    for _ in range(num_blocks):
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


def exit_flow(x, y):
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

    z = tf.math.add(x, y)  # fuse the two channels
    z = GlobalAveragePooling2D()(z)
    z = Dense(2, activation='linear')(z)

    return z


inputs = Input(shape=(224, 224, 3))
outputs_x1, outputs_y1 = entry_flow(inputs)
outputs_x2, outputs_y2 = middle_flow(outputs_x1, outputs_y1)
outputs = exit_flow(outputs_x2, outputs_y2)
xception = Model(inputs, outputs)


def load_and_preprocess_from_path_label(image, label):
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
Final_CM = np.mat(np.zeros((4, 4)))
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
    train_db_right = train_dataset_right.shuffle(2000).map(load_and_preprocess_from_path_label).batch(5)

    test_dataset_right = tf.data.Dataset.from_tensor_slices((test_image_right, test_label_right))
    test_db_right = test_dataset_right.shuffle(2000).map(load_and_preprocess_from_path_label).batch(5)

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

    epochs = 10  # 35
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

        #     with CV_summary_writer.as_default():
        #         tf.summary.scalar('train-CrossEntropy', float(loss), step=epoch)
        #         tf.summary.scalar('train-Accuracy', float(train_accuracy.result() * 100), step=epoch)
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

        #     with CV_summary_writer.as_default():
        #         tf.summary.scalar('train-CrossEntropy', float(loss), step=epoch)
        #         tf.summary.scalar('train-Accuracy', float(train_accuracy.result() * 100), step=epoch)
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
        for j in range(len(test_pred_right)):
            if test_GT_left[j].numpy() == 0 and test_GT_right[j].numpy() == 0:
                ground_truth.append(0)
            elif test_GT_left[j].numpy() == 1 and test_GT_right[j].numpy() == 0:
                ground_truth.append(1)
            elif test_GT_left[j].numpy() == 0 and test_GT_right[j].numpy() == 1:
                ground_truth.append(2)
            elif test_GT_left[j].numpy() == 1 and test_GT_right[j].numpy() == 1:
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

        print('\nClassification Report\n')
        print(classification_report(ground_truth, prediction, labels=range(4),
                                    target_names=['Normal', 'Left Abnormal', 'Right Abnormal', 'Abnormal']))

        print("Confusion Matrix")
        print(CM)

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


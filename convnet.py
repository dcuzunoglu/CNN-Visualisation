"""
CNN Training script for Oxford 17 flowers dataset

Note: The first 7 classes are removed from training and prediction.
Therefore, images 1-560 belong to classes that the model does not recognize.

Following command line arguments can be used while calling this script
    - train -> initiate model training and save model at the end of training
    - test -> load saved model and do prediction on the whole test set
    - activation $IMAGENAME -> run the given image through first convolution layer, then predict image and save
                               visualised convolution activations as a png to disk

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import sys
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread

from constants import IMAGE_SIZE, TRAIN_SET_FILE_PATH, TEST_SET_FILE_PATH, TRAIN_SET_LABELS_FILE_PATH, \
    TEST_SET_LABELS_FILE_PATH, METRICS_FILE_PATH, MODEL_FILE_PATH, CM_FILE_MATH, MODEL_DIR, MODEL_META_FILE_PATH, \
    TOTAL_NUM_CLASSES, REMOVED_NUM_CLASSES, ALL_LABELS_CSV_FILE_PATH, FLOWERS_RESIZED_DIR, FLOWER_NAMES, \
    ACTIVATION_IMAGES_DIR, FLOWERS_ORIGINAL_DIR, STATIC_IMAGES_DIR

# CNN constants
NUM_INPUT_CHANNELS = 3

LAYER_1_FILTER_SIZE = 32
LAYER_2_FILTER_SIZE = 64

KERNEL_SIZE = [2, 2]
PADDING = "same"
ACTIVATION_FUNCTION = tf.nn.relu

POOL_SIZE = [2, 2]
STRIDES = 2

FC_NUM_NEURONS = 1024
DROPOUT_RATE = 0.05

NUM_CLASSES = TOTAL_NUM_CLASSES - REMOVED_NUM_CLASSES

# Training parameters
LEARNING_RATE = 0.1
BATCH_SIZE = 100
NUM_EPOCHS = 100


def cnn_model(input_x, training_phase):
    """

    :param input_x:
    :param training_phase:
    :return:
    """
    x_norm = tf.layers.batch_normalization(input_x, training=training_phase)
    input_layer = tf.reshape(x_norm, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_INPUT_CHANNELS])

    # Convolutional layer 1
    # Computes LAYER_1_FILTER_SIZE features using a KERNEL_SIZE filter with ACTIVATION_FUNCTION activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_INPUT_CHANNELS]
    # Output Tensor Shape: [batch_size, IMAGE_SIZE, IMAGE_SIZE, LAYER_1_FILTER_SIZE]
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=LAYER_1_FILTER_SIZE,
        kernel_size=KERNEL_SIZE,
        padding=PADDING,
        activation=ACTIVATION_FUNCTION, name="conv_1")

    # Pooling Layer #1
    # First max pooling layer with a KERNEL_SIZE filter and stride of STRIDES
    # Input Tensor Shape: [batch_size, IMAGE_SIZE, IMAGE_SIZE, LAYER_1_FILTER_SIZE]
    # Output Tensor Shape: [batch_size, IMAGE_SIZE / 2, IMAGE_SIZE / 2, LAYER_1_FILTER_SIZE]
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=POOL_SIZE,
        strides=STRIDES)

    # Convolutional Layer #2
    # Computes 64 features using a KERNEL_SIZE filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, IMAGE_SIZE / 2, IMAGE_SIZE / 2, LAYER_1_FILTER_SIZE]
    # Output Tensor Shape: [batch_size, IMAGE_SIZE / 2, IMAGE_SIZE / 2, LAYER_2_FILTER_SIZE]
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=LAYER_2_FILTER_SIZE,
        kernel_size=KERNEL_SIZE,
        padding=PADDING,
        activation=ACTIVATION_FUNCTION, name="conv_2")

    # Pooling Layer #2
    # Second max pooling layer with a KERNEL_SIZE filter and stride of STRIDES
    # Input Tensor Shape: [batch_size, IMAGE_SIZE / 2, IMAGE_SIZE / 2, LAYER_2_FILTER_SIZE]
    # Output Tensor Shape: [batch_size, IMAGE_SIZE / 4, IMAGE_SIZE / 4, LAYER_2_FILTER_SIZE]
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=POOL_SIZE,
        strides=STRIDES)

    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, IMAGE_SIZE / 4, IMAGE_SIZE / 4, LAYER_2_FILTER_SIZE]
    # Output Tensor Shape: [batch_size, IMAGE_SIZE / 4 * IMAGE_SIZE / 4 * LAYER_2_FILTER_SIZE]
    pool2_flattened = tf.reshape(pool2, [-1, pool2.shape[1] * pool2.shape[2] * pool2.shape[3]])

    # Dense Layer
    # Densely connected layer with FC_NUM_NEURONS neurons
    # Input Tensor Shape: [batch_size, IMAGE_SIZE / 4 * IMAGE_SIZE / 4 * LAYER_2_FILTER_SIZE]
    # Output Tensor Shape: [batch_size, FC_NUM_NEURONS]
    dense = tf.layers.dense(inputs=pool2_flattened, units=FC_NUM_NEURONS, activation=ACTIVATION_FUNCTION)

    # Add dropout operation; 1 - DROPOUT_RATE probability that element will be kept
    dropout = tf.layers.dropout(
        inputs=dense, rate=DROPOUT_RATE)

    # Logits layer
    # Input Tensor Shape: [batch_size, FC_NUM_NEURONS]
    # Output Tensor Shape: [batch_size, NUM_CLASSES]
    logits = tf.layers.dense(inputs=dropout, units=NUM_CLASSES)

    classes = tf.argmax(input=logits, axis=1, name="prediction_results")  # also to predict when model restored

    return {"logits": logits, "classes": classes}


def train_and_save_network():
    """

    :return:
    """

    # ============================= DATASET PREPARATION ========================================
    # Load training and eval data
    train_data = np.load(TRAIN_SET_FILE_PATH)
    train_labels = np.load(TRAIN_SET_LABELS_FILE_PATH)
    train_labels_one_hot = np.zeros((len(train_labels), NUM_CLASSES))

    test_data = np.load(TEST_SET_FILE_PATH)
    test_labels = np.load(TEST_SET_LABELS_FILE_PATH)
    test_labels_one_hot = np.zeros((len(test_labels), NUM_CLASSES))

    # One hot encode labels
    for i in range(len(train_labels)):
        train_labels_one_hot[i][int(train_labels[i]) - (REMOVED_NUM_CLASSES + 1)] = 1
    for i in range(len(test_labels)):
        test_labels_one_hot[i][int(test_labels[i]) - (REMOVED_NUM_CLASSES + 1)] = 1

    # Shuffle
    randomize = np.arange(0, len(train_data), 1)
    np.random.shuffle(randomize)
    dataset_shuffled = []
    labels_shuffled = []

    for i in range(len(randomize)):
        dataset_shuffled.append(train_data[randomize[i]])
        labels_shuffled.append(train_labels_one_hot[randomize[i]])

    train_data = dataset_shuffled
    train_labels_one_hot = labels_shuffled

    # ========================= TF OPERATION DEFITIONS =========================================

    # Variables
    x_placeholder = tf.placeholder('float', [None, IMAGE_SIZE, IMAGE_SIZE, 3], name='x_placeholder')
    y_placeholder = tf.placeholder('float', name='y_placeholder')
    is_train = tf.placeholder(tf.bool, name="is_train")

    # Forward and back propagation
    prediction = cnn_model(x_placeholder, is_train)
    logits = prediction["logits"]
    classes = prediction["classes"]
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_placeholder)
    cross_entropy_avg = tf.reduce_mean(cross_entropy_loss)
    optimizer = tf.train.AdamOptimizer().minimize(cross_entropy_avg)

    # Performance metric operations
    _, accuracy_op = tf.metrics.accuracy(labels=tf.argmax(y_placeholder, 1), predictions=classes,
                                         name="accuracy_calculator")
    _, recall_op = tf.metrics.recall(labels=tf.argmax(y_placeholder, 1), predictions=classes,
                                     name="recall_calculator")
    _, precision_op = tf.metrics.precision(labels=tf.argmax(y_placeholder, 1), predictions=classes,
                                           name="precision_calculator")
    confusion_matrix_op = tf.confusion_matrix(labels=tf.argmax(y_placeholder, 1), predictions=classes,
                                              name="cm_calculator")

    # ========================= TRAINING SESSION ===============================================
    metrics_per_epoch = []
    num_batches = int(len(train_data) / BATCH_SIZE)

    # Start session, initialize variables and model saver
    with tf.Session() as sess:
        with tf.device('/cpu:0'):

            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            saver = tf.train.Saver()

            # For every epoch
            for epoch in range(NUM_EPOCHS):
                print("\nEpoch {}".format(epoch))
                epoch_loss = 0
                epoch_train_accuracy = 0

                # For every batch
                for i in range(num_batches):

                    # Backpropagate and record losses and accuracy for this batch
                    current_batch = train_data[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
                    current_labels = train_labels_one_hot[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

                    _, loss = sess.run([optimizer, cross_entropy_avg], feed_dict={x_placeholder: current_batch,
                                                                                  y_placeholder: current_labels,
                                                                                  is_train: 1})

                    batch_accuracy = sess.run(accuracy_op, feed_dict={x_placeholder: current_batch,
                                                                      y_placeholder: current_labels,
                                                                      is_train: 0})

                    # Add to total loss and accuracy
                    epoch_loss += loss
                    epoch_train_accuracy += batch_accuracy

                # At the end of each epoch, calculate performance metrics over test set and record
                train_accuracy = epoch_train_accuracy / num_batches
                test_accuracy, test_precision, test_recall = sess.run([accuracy_op, precision_op, recall_op],
                                                                      feed_dict={x_placeholder: test_data,
                                                                                 y_placeholder: test_labels_one_hot,
                                                                                 is_train: 0})
                test_f1 = 2 * (test_recall * test_precision) / (test_recall + test_precision)

                # Columns on the metric list are:
                # TRAIN LOSS, TRAIN ACCURACY, TEST ACCURACY, TEST PRECISION, TEST RECALL, TEST F1
                metrics_per_epoch.append([epoch_loss, train_accuracy, test_accuracy,
                                          test_precision, test_recall, test_f1])

                print("\nEpoch {}/{} completed.\nTrain Loss: {}\nTrain Accuracy: {}\n"
                      "Test Accuracy: {}\nTest Precision: {}\n"
                      "Test Recall: {}\nTest F1: {}".format(epoch, NUM_EPOCHS, epoch_loss, train_accuracy,
                                                            test_accuracy, test_precision, test_recall, test_f1))

                # Save the model at the end of training
                if epoch == NUM_EPOCHS - 1:

                    # Compute confusion matrix
                    confusion_matrix = sess.run(confusion_matrix_op, feed_dict={x_placeholder: test_data,
                                                                                y_placeholder: test_labels_one_hot,
                                                                                is_train: 0})

                    # Save model, metrics, and confusion matrix
                    save_path = saver.save(sess, MODEL_FILE_PATH)
                    np.savetxt(METRICS_FILE_PATH, np.asarray(metrics_per_epoch), delimiter=',')
                    np.savetxt(CM_FILE_MATH, confusion_matrix, delimiter=',')

                    print("Saving model in {}.\n".format(save_path))
                    print(confusion_matrix)


def calculate_activations_for_image_and_predict(image_name, which_layer=1, from_flask=False):
    """
    Load model from given model_path and predict test set
    :param image_name:
    :param which_layer:
    :param from_flask:
    :return:
    """
    # Read in image for prediction
    image = imread(FLOWERS_RESIZED_DIR + image_name)
    image_original = imread(FLOWERS_ORIGINAL_DIR + image_name)

    # Read in labels (to show prediction results on graph)
    all_labels = np.loadtxt(ALL_LABELS_CSV_FILE_PATH, delimiter=',')
    current_label = int(all_labels[int(image_name[-8:-4]) - 1])
    label_text = FLOWER_NAMES[current_label - 1]

    if current_label < 8:  # If our model doesn't know this class
        unrecognisable = 1
        label_text = label_text + " (UNRECOGNISABLE CLASS)"
    else:
        unrecognisable = 0

    # Reset graph and initialize session
    tf.reset_default_graph()

    with tf.Session() as sess:

        # Import graph structure and restore last checkpoint
        loader = tf.train.import_meta_graph(MODEL_META_FILE_PATH)
        loader.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
        print("Successfully loaded model from {}".format(MODEL_FILE_PATH))

        # Initialize graph and tensors
        graph = tf.get_default_graph()
        x_placeholder = graph.get_tensor_by_name("x_placeholder:0")
        is_train = graph.get_tensor_by_name("is_train:0")

        # Extract the prediction results operation and given convnet layer
        classes = graph.get_tensor_by_name("prediction_results:0")
        conv_layer = graph.get_tensor_by_name("conv_{}/Conv2D:0".format(which_layer, which_layer))

        # Reshape image and feed into the conv layer to get activations, and accuracy layer to get predictions
        image_tensor = np.reshape(image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
        activations, prediction = sess.run([conv_layer, classes], feed_dict={x_placeholder: image_tensor,
                                                                             is_train: 0})

        # Find label that prediction corresponds to
        if unrecognisable:
            prediction_text = "N/A"
        else:
            prediction_text = FLOWER_NAMES[prediction[0] + REMOVED_NUM_CLASSES]

        # Define graph parameters
        filters = activations.shape[3]
        n_columns = 4
        n_rows = math.ceil(filters / n_columns)
        title = "Image class: {}\nPredicted class: {}".format(label_text, prediction_text)

        fig = plt.figure(1, figsize=(32, 18))
        fig.suptitle(title, fontsize=24, color='r', horizontalalignment='right', verticalalignment='top')
        gs = gridspec.GridSpec(n_rows, n_columns * 2)

        # Plot original image on one half of graph
        plt.subplot(gs[:, :n_columns])
        plt.axis('off')  # remove axes
        plt.imshow(image_original, interpolation="nearest", aspect='auto')

        # Then plot all filters on remainder of graph
        filter_index = 0

        for i in range(n_rows):
            for j in range(n_columns):
                plt.subplot(gs[i, n_columns + j])
                plt.axis('off')  # remove axes
                plt.imshow(activations[0, :, :, filter_index], interpolation="nearest", cmap="gray")
                filter_index += 1

        fig.subplots_adjust(hspace=0.01, wspace=1, top=1, bottom=0, right=1, left=0)
        plt.tight_layout()

        # If request by flask, save to static and return file name, if not show graph and save to default dir
        if from_flask:
            image_path = STATIC_IMAGES_DIR + "0" + image_name[:-4] + ".jpg"
            fig.savefig(image_path, bbox_inches='tight', pad_inches=0, type="jpg")
            fig.savefig(ACTIVATION_IMAGES_DIR + image_name[:-4] + ".png", bbox_inches='tight', pad_inches=0)
            return image_path
        else:
            plt.show()
            fig.savefig(ACTIVATION_IMAGES_DIR + image_name[:-4] + ".png", bbox_inches='tight', pad_inches=0)


def predict_test_set():
    """
    Runs the convnet model on the test set and outputs performance metrics
    :return:
    """
    # Prepare test data
    test_data = np.load(TEST_SET_FILE_PATH)
    test_labels = np.load(TEST_SET_LABELS_FILE_PATH)
    test_labels_one_hot = np.zeros((len(test_labels), NUM_CLASSES))

    for i in range(len(test_labels)):
        test_labels_one_hot[i][int(test_labels[i]) - (REMOVED_NUM_CLASSES + 1)] = 1

    # Reset graph and initialize session
    tf.reset_default_graph()

    with tf.Session() as sess:

        # Import graph structure and restore last checkpoint
        loader = tf.train.import_meta_graph(MODEL_META_FILE_PATH)
        loader.restore(sess, tf.train.latest_checkpoint(MODEL_DIR))
        print("Successfully loaded model from {}".format(MODEL_FILE_PATH))

        # Initialize graph and tensors
        graph = tf.get_default_graph()
        x_placeholder = graph.get_tensor_by_name("x_placeholder:0")
        y_placeholder = graph.get_tensor_by_name("y_placeholder:0")
        is_train = graph.get_tensor_by_name("is_train:0")

        # Extract the prediction results operation
        classes = graph.get_tensor_by_name("prediction_results:0")

        # Run performance operations, and print results
        correct = tf.equal(classes, tf.argmax(y_placeholder, 1))
        accuracy_op = tf.reduce_mean(tf.cast(correct, 'float'))
        accuracy = sess.run(accuracy_op, feed_dict={x_placeholder: test_data, y_placeholder: test_labels_one_hot,
                                                    is_train: 0})

        print("Test Accuracy: {}".format(accuracy))


# ============================================ MAIN ================================================================
if __name__ == "__main__":

    # Check command line arguments to run the appropriate function
    if sys.argv[1] == "train":
        train_and_save_network()
    elif sys.argv[1] == "test":
        predict_test_set()
    elif sys.argv[1] == "activation" and len(sys.argv) == 3:
        all_images = os.listdir(FLOWERS_RESIZED_DIR)

        if sys.argv[2] not in all_images:
            print("{} not found in {}. Exiting.".format(sys.argv[2], FLOWERS_RESIZED_DIR))
            sys.exit(-1)

        calculate_activations_for_image_and_predict(sys.argv[2])
    else:
        print("convnet.py train -> initiates CNN train and saves model\n"
              "convnet.py test -> loads saved model and predicts whole test set\n"
              "convnet.py activation $IMAGENAME -> loads $IMAGENAME from {} and runs through first convolution layer"
              "to get activations. Then predicts $IMAGENAME and saves visualised "
              "activations to disk.".format(FLOWERS_RESIZED_DIR))


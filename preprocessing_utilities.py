"""
Helper functions

"""
import os

from scipy import io
from skimage.util import random_noise
import numpy as np
import cv2

from constants import DATASET_SPLITS_MAT_FILE_PATH, ALL_LABELS_CSV_FILE_PATH, \
    TRAIN_FILES_CSV_FILE_PATH, TRAIN_LABELS_CSV_FILE_PATH, TEST_LABELS_CSV_FILE_PATH,\
    VAL_FILES_CSV_FILE_PATH, TEST_FILES_CSV_FILE_PATH, FLOWERS_ORIGINAL_DIR, FLOWERS_RESIZED_DIR, IMAGE_SIZE, \
    TRAIN_SET_FILE_PATH, TEST_SET_FILE_PATH, TRAIN_SET_LABELS_FILE_PATH, \
    TEST_SET_LABELS_FILE_PATH, REMOVED_NUM_CLASSES, TOTAL_NUM_CLASSES


def augment_images(original_img_dir, train_files_path, dataset_no):
    """

    :param original_img_dir:
    :param train_files_path:
    :param dataset_no
    :return:
    """

    # Get all files corresponding to given dataset split from train_files
    # Convert file number into string, then add appropriate number of zeros to
    # form proper file name (total number of digits in file name is 5)
    files_list = np.loadtxt(train_files_path, delimiter=',')

    if dataset_no != -1:
        files_list = files_list[:, dataset_no]

    file_name_list = []

    for i in range(len(files_list)):

        file_name = str(int(files_list[i]))
        num_zeros = 4 - len(file_name)

        for j in range(num_zeros):
            file_name = "0" + file_name

        file_name = "image_" + file_name + ".jpg"
        file_name_list.append(file_name)
    file_name_list = sorted(file_name_list)

    for image_file in file_name_list:
        original_image = cv2.imread(original_img_dir + image_file)
        augmented_image = random_noise(original_image, mode='gaussian') * 256
        file_path = original_img_dir + image_file[:-4] + "_aug.jpg"
        cv2.imwrite(file_path, augmented_image)
        print("Augmented and saved in {}".format(file_path))


def mat_to_csv(input_file_path, output_file_path, channels):
    """
    Converts the given .mat file into .csv and saves
    There are 3 different dataset splits, therefore resulting CSV has three column with each
    column corresponding to a split
    :param input_file_path:
    :param output_file_path:
    :param channels:
    :return:
    """
    input_file = io.loadmat(input_file_path)

    output_array = input_file[channels[0]]
    channels = channels[1:]

    for channel in sorted(channels):
        output_array = np.concatenate((output_array, input_file[channel]))

    np.savetxt(output_file_path, np.transpose(output_array), delimiter=',')
    print("Saved {}".format(output_file_path))


def concat_train_and_val(train_file_path, val_file_path):
    """

    :param train_file_path:
    :param val_file_path:
    :return:
    """
    train_files = np.loadtxt(train_file_path, delimiter=',')
    val_files = np.loadtxt(val_file_path, delimiter=',')

    combined = np.concatenate((train_files, val_files), axis=0)
    np.savetxt(train_file_path, combined, delimiter=',')


def create_all_labels_csv(output_file_path, num_classes, num_examples_per_class):
    """
    Images are ordered for labelling and there are 80 examples per class. For example images 1-80 are class 1,
    images 81-160 are class 2 and so forth. Creates a CSV file with labels for all images
    :param output_file_path:
    :param num_classes:
    :param num_examples_per_class:
    :return:
    """

    all_labels = np.zeros(num_classes * num_examples_per_class)

    for i in range(num_classes):
        all_labels[i * num_examples_per_class:(i + 1) * num_examples_per_class] = i + 1

    np.savetxt(output_file_path, all_labels, delimiter=',')


def resize_images(original_img_dir, resized_img_dir, new_size):
    """
    Resizes all images in original image directory to the given new size, and saves
    them in the resized image directory
    :param original_img_dir:
    :param resized_img_dir:
    :param new_size
    :return:
    """

    all_files = os.listdir(original_img_dir)

    for image_file in all_files:
        original_image = cv2.imread(original_img_dir + image_file)
        print(image_file)
        resized_image = cv2.resize(original_image, (new_size, new_size))
        cv2.imwrite(resized_img_dir + image_file, resized_image)
        print("Resized and saved in {}".format(resized_img_dir + image_file))


def create_label_csv(all_labels_file_path, files_list_file_path, output_file_path):
    """
    Creates CSV files that have labels for the given dataset file list. Note that each file list has
    3 dataset splits, so the output CSV file from this function also has three columns of labels corresponding to
    each split
    :param all_labels_file_path:
    :param files_list_file_path:
    :param output_file_path:
    :return:
    """

    file_labels = []

    # Read in all labels and list of data files
    all_labels = np.loadtxt(all_labels_file_path)
    dataset_file_array = np.loadtxt(files_list_file_path, delimiter=',')

    for i in range(dataset_file_array.shape[1]):
        dataset = dataset_file_array[:, i]
        dataset_file_labels = []

        for data_file in dataset:
            dataset_file_labels.append(all_labels[int(data_file - 1)])

        file_labels.append(dataset_file_labels)

    np.savetxt(output_file_path, np.transpose(file_labels), delimiter=',')
    print("Saved labels in {}".format(output_file_path))


def create_numpy_dataset_for_cnn(files_list_file_path, files_dir_path, labels_file_path, data_output_file_path,
                                 labels_output_file_path, dataset_no):
    """
    Creates a numpy dataset for the given dataset number (there are 3 splits) using
    :param files_list_file_path: this is the dataset files list file
    :param files_dir_path: this is where the images are
    :param labels_file_path: this is the labels for dataset files
    :param data_output_file_path: this has the data matrices
    :param labels_output_file_path: this has the labels for corresponding matrices
    :param dataset_no:
    :return:
    """
    dataset = []
    dataset_labels = []

    # Read in files list and labels
    files_list = np.loadtxt(files_list_file_path, delimiter=',')
    all_labels = np.loadtxt(labels_file_path, delimiter=',')

    # Only use the specified dataset number
    if dataset_no != -1:
        files_list = files_list[:, dataset_no]
        all_labels = all_labels[:, dataset_no]

    # Loop over files list, read in images and labels into dataset list
    for i in range(len(files_list)):

        # Convert file number into string, then add appropriate number of zeros to
        # form proper file name (total number of digits in file name is 5)
        file_name = str(int(files_list[i]))
        num_zeros = 4 - len(file_name)

        for j in range(num_zeros):
            file_name = "0" + file_name

        file_name = "image_" + file_name + ".jpg"

        # if "train_files" in files_list_file_path:
        #     file_name_aug = file_name[:-4] + "_aug.jpg"
        #     image_aug = cv2.imread(files_dir_path + file_name_aug, cv2.IMREAD_COLOR)
        #     dataset.append(np.array(image_aug))
        #     dataset_labels.append(all_labels[i])

        # Read file, and append with label into dataset list
        image = cv2.imread(files_dir_path + file_name, cv2.IMREAD_COLOR)
        dataset.append(np.array(image))
        dataset_labels.append(all_labels[i])

    # Shuffle and save
    randomize = np.arange(0, len(dataset), 1)
    np.random.shuffle(randomize)
    dataset_shuffled = []
    labels_shuffled = []

    for i in range(len(randomize)):
        dataset_shuffled.append(dataset[randomize[i]])
        labels_shuffled.append(dataset_labels[randomize[i]])

    np.save(data_output_file_path, dataset_shuffled)
    np.save(labels_output_file_path, labels_shuffled)
    print("Saved dataset into {}, size {}".format(data_output_file_path, len(dataset_shuffled)))


def remove_n_classes(train_files_list_path, train_labels_list_path, test_files_list_path,
                     test_labels_list_path, num_classes, num_examples_per_class, dataset_num):
    """

    :param train_files_list_path:
    :param train_labels_list_path:
    :param test_files_list_path:
    :param test_labels_list_path:
    :param num_classes:
    :param num_examples_per_class:
    :param dataset_num:
    :return:
    """
    train_files = np.loadtxt(train_files_list_path, delimiter=',')
    train_files = train_files[:, dataset_num]

    test_files = np.loadtxt(test_files_list_path, delimiter=',')
    test_files = test_files[:, dataset_num]

    train_labels = np.loadtxt(train_labels_list_path, delimiter=',')
    train_labels = train_labels[:, dataset_num]

    test_labels = np.loadtxt(test_labels_list_path, delimiter=',')
    test_labels = test_labels[:, dataset_num]

    rows_to_remove = np.arange(0, num_classes*num_examples_per_class + 1)

    for row in rows_to_remove:

        if row in train_files:

            index = np.where(train_files == row)
            index = index[0]
            train_files = np.delete(train_files, index)
            train_labels = np.delete(train_labels, index)
        elif row in test_files:
            index = np.where(test_files == row)
            index = index[0]
            test_files = np.delete(test_files, index)
            test_labels = np.delete(test_labels, index)

    np.savetxt(train_labels_list_path, train_labels)
    np.savetxt(train_files_list_path, train_files)
    np.savetxt(test_labels_list_path, test_labels)
    np.savetxt(test_files_list_path, test_files)
    print("Removed {} classes and saved.".format(num_classes))


if __name__ == "__main__":

    # Convert original dataset splits to CSVs
    mat_to_csv(DATASET_SPLITS_MAT_FILE_PATH, TRAIN_FILES_CSV_FILE_PATH, ['trn1', 'trn2', 'trn3'])
    mat_to_csv(DATASET_SPLITS_MAT_FILE_PATH, VAL_FILES_CSV_FILE_PATH, ['val1', 'val2', 'val3'])
    mat_to_csv(DATASET_SPLITS_MAT_FILE_PATH, TEST_FILES_CSV_FILE_PATH, ['tst1', 'tst2', 'tst3'])
    concat_train_and_val(TRAIN_FILES_CSV_FILE_PATH, VAL_FILES_CSV_FILE_PATH)

    # Create labels CSV
    create_all_labels_csv(ALL_LABELS_CSV_FILE_PATH, TOTAL_NUM_CLASSES, 80)

    # Create label files for train and test sets
    create_label_csv(ALL_LABELS_CSV_FILE_PATH, TRAIN_FILES_CSV_FILE_PATH, TRAIN_LABELS_CSV_FILE_PATH)
    create_label_csv(ALL_LABELS_CSV_FILE_PATH, TEST_FILES_CSV_FILE_PATH, TEST_LABELS_CSV_FILE_PATH)

    # Remove n classes
    remove_n_classes(TRAIN_FILES_CSV_FILE_PATH, TRAIN_LABELS_CSV_FILE_PATH, TEST_FILES_CSV_FILE_PATH,
                     TEST_LABELS_CSV_FILE_PATH, REMOVED_NUM_CLASSES, 80, 0)

    # Augment files in training set
    # augment_images(FLOWERS_ORIGINAL_DIR, TRAIN_FILES_CSV_FILE_PATH, -1)

    # Resize images
    resize_images(FLOWERS_ORIGINAL_DIR, FLOWERS_RESIZED_DIR, IMAGE_SIZE)

    # Create numpy array training, and test sets
    create_numpy_dataset_for_cnn(TRAIN_FILES_CSV_FILE_PATH, FLOWERS_RESIZED_DIR,
                                 TRAIN_LABELS_CSV_FILE_PATH, TRAIN_SET_FILE_PATH, TRAIN_SET_LABELS_FILE_PATH, -1)
    create_numpy_dataset_for_cnn(TEST_FILES_CSV_FILE_PATH, FLOWERS_RESIZED_DIR,
                                 TEST_LABELS_CSV_FILE_PATH, TEST_SET_FILE_PATH, TEST_SET_LABELS_FILE_PATH, -1)

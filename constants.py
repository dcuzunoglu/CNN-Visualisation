"""
Script for directory paths and other constants

"""
import os

# Directory paths
CURRENT_DIR = os.getcwd()
FLOWERS_ORIGINAL_DIR = CURRENT_DIR + "/flower-images/"
FLOWERS_RESIZED_DIR = CURRENT_DIR + "/flower-images-resized/"
CSV_DIR = CURRENT_DIR + "/CSV-files/"
MODEL_DIR = CURRENT_DIR + "/models/"
DATASET_DIR = CURRENT_DIR + "/datasets/"
ACTIVATION_IMAGES_DIR = CURRENT_DIR + "/activation-images/"
STATIC_IMAGES_DIR = CURRENT_DIR + "/static/"

# Main directory file paths
TRAIN_SET_FILE_PATH = DATASET_DIR + "train.npy"
TRAIN_SET_LABELS_FILE_PATH = DATASET_DIR + "train_labels.npy"
TEST_SET_FILE_PATH = DATASET_DIR + "test.npy"
TEST_SET_LABELS_FILE_PATH = DATASET_DIR + "test_labels.npy"

# Models directory file paths
METRICS_FILE_PATH = MODEL_DIR + "metrics.csv"
CM_FILE_MATH = MODEL_DIR + "confusion_matrix.csv"
MODEL_FILE_PATH = MODEL_DIR + "cnn.cpkt"
MODEL_META_FILE_PATH = MODEL_DIR + "cnn.cpkt.meta"

# CSV directory file paths
DATASET_SPLITS_MAT_FILE_PATH = CSV_DIR + "datasplits.mat"
ALL_LABELS_CSV_FILE_PATH = CSV_DIR + "all_labels.csv"
TRAIN_FILES_CSV_FILE_PATH = CSV_DIR + "train_files.csv"
TRAIN_LABELS_CSV_FILE_PATH = CSV_DIR + "train_labels.csv"
VAL_FILES_CSV_FILE_PATH = CSV_DIR + "validation_files.csv"
TEST_FILES_CSV_FILE_PATH = CSV_DIR + "test_files.csv"
TEST_LABELS_CSV_FILE_PATH = CSV_DIR + "test_labels.csv"

IMAGE_SIZE = 128
TOTAL_NUM_CLASSES = 17
REMOVED_NUM_CLASSES = 7  # First REMOVED_NUM_CLASSES are removed
FLOWER_NAMES = ["Daffodil", "Snowdrop", "Lily Valley", "Bluebell", "Crocus", "Iris", "Tigerlily", "Tulip", "Fritillary",
                "Sunflower", "Daisy", "Colts' Foot", "Dandelion", "Cowslip", "Buttercup", "Windflower", "Pansy"]



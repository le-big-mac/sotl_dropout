import math
import numpy as np
import argparse
import torch
import train

parser = argparse.ArgumentParser()

parser.add_argument('--dir', '-d', required=True, help='Name of the UCI Dataset directory. Eg: bostonHousing')
parser.add_argument('--epochx', '-e', default=500, type=int, help='Multiplier for the number of epochs for training.')
parser.add_argument('--hidden', '-nh', default=2, type=int, help='Number of hidden layers for the neural net')
parser.add_argument('--normalize', '-n', default=True, type=bool, help='Normalize training features')
parser.add_argument('--normalize_labels', '-nl', default=True, type=bool,
                    help='Normalize labels (useful for additive noise)')

args = parser.parse_args()

data_directory = args.dir
epochs_multiplier = args.epochx
num_hidden_layers = args.hidden
normalize = args.normalize
normalize_labels = args.normalize_labels

# We delete previous results

from subprocess import call

_RESULTS_VALIDATION_LL = "./UCI_Datasets/" + data_directory + "/results/validation_ll_" + str(
    epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_VALIDATION_RMSE = "./UCI_Datasets/" + data_directory + "/results/validation_rmse_" + str(
    epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_VALIDATION_MC_RMSE = "./UCI_Datasets/" + data_directory + "/results/validation_MC_rmse_" + str(
    epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"

_RESULTS_TEST_LL = "./UCI_Datasets/" + data_directory + "/results/test_ll_" + str(
    epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_TAU = "./UCI_Datasets/" + data_directory + "/results/test_tau_" + str(
    epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_RMSE = "./UCI_Datasets/" + data_directory + "/results/test_rmse_" + str(
    epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_MC_RMSE = "./UCI_Datasets/" + data_directory + "/results/test_MC_rmse_" + str(
    epochs_multiplier) + "_xepochs_" + str(num_hidden_layers) + "_hidden_layers.txt"
_RESULTS_TEST_LOG = "./UCI_Datasets/" + data_directory + "/results/log_" + str(epochs_multiplier) + "_xepochs_" + str(
    num_hidden_layers) + "_hidden_layers.txt"

_DATA_DIRECTORY_PATH = "./UCI_Datasets/" + data_directory + "/data/"
_DROPOUT_RATES_FILE = _DATA_DIRECTORY_PATH + "dropout_rates.txt"
_TAU_VALUES_FILE = _DATA_DIRECTORY_PATH + "tau_values.txt"
_DATA_FILE = _DATA_DIRECTORY_PATH + "data.txt"
_HIDDEN_UNITS_FILE = _DATA_DIRECTORY_PATH + "n_hidden.txt"
_EPOCHS_FILE = _DATA_DIRECTORY_PATH + "n_epochs.txt"
_INDEX_FEATURES_FILE = _DATA_DIRECTORY_PATH + "index_features.txt"
_INDEX_TARGET_FILE = _DATA_DIRECTORY_PATH + "index_target.txt"
_N_SPLITS_FILE = _DATA_DIRECTORY_PATH + "n_splits.txt"


def _get_index_train_test_path(split_num, train=True):
    """
       Method to generate the path containing the training/test split for the given
       split number (generally from 1 to 20).
       @param split_num      Split number for which the data has to be generated
       @param train          Is true if the data is training data. Else false.
       @return path          Path of the file containing the requried data
    """
    if train:
        return _DATA_DIRECTORY_PATH + "index_train_" + str(split_num) + ".txt"
    else:
        return _DATA_DIRECTORY_PATH + "index_test_" + str(split_num) + ".txt"


# print("Removing existing result files...")
# call(["rm", _RESULTS_VALIDATION_LL])
# call(["rm", _RESULTS_VALIDATION_RMSE])
# call(["rm", _RESULTS_VALIDATION_MC_RMSE])
# call(["rm", _RESULTS_TEST_LL])
# call(["rm", _RESULTS_TEST_TAU])
# call(["rm", _RESULTS_TEST_RMSE])
# call(["rm", _RESULTS_TEST_MC_RMSE])
# call(["rm", _RESULTS_TEST_LOG])
# print("Result files removed.")

# We fix the random seed

np.random.seed(1)
torch.manual_seed(1)

print("Loading data and other hyperparameters...")
# We load the data

data = np.loadtxt(_DATA_FILE)

# We load the number of hidden units

n_hidden = np.loadtxt(_HIDDEN_UNITS_FILE).tolist()

# We load the number of training epochs

n_epochs = np.loadtxt(_EPOCHS_FILE).tolist()

# We load the indexes for the features and for the target

index_features = np.loadtxt(_INDEX_FEATURES_FILE)
index_target = np.loadtxt(_INDEX_TARGET_FILE)

X = data[:, [int(i) for i in index_features.tolist()]]
y = data[:, int(index_target.tolist())]

# We iterate over the training test splits

n_splits = np.loadtxt(_N_SPLITS_FILE)
print("Done.")

errors, MC_errors, lls = [], [], []
for split in range(int(n_splits)):

    # We load the indexes of the training and test sets
    print('Loading file: ' + _get_index_train_test_path(split, train=True))
    print('Loading file: ' + _get_index_train_test_path(split, train=False))
    index_train = np.loadtxt(_get_index_train_test_path(split, train=True))
    index_test = np.loadtxt(_get_index_train_test_path(split, train=False))

    X_train = X[[int(i) for i in index_train.tolist()]]
    y_train = y[[int(i) for i in index_train.tolist()]]

    X_test = X[[int(i) for i in index_test.tolist()]]
    y_test = y[[int(i) for i in index_test.tolist()]]

    if normalize:
        std_X_train = np.std(X_train, 0)
        std_X_train[std_X_train == 0] = 1
        mean_X_train = np.mean(X_train, 0)
    else:
        std_X_train = np.ones(X_train.shape[1])
        mean_X_train = np.zeros(X_train.shape[1])

    X_train = (X_train - np.full(X_train.shape, mean_X_train)) / np.full(X_train.shape, std_X_train)
    X_test = (X_test - np.full(X_test.shape, mean_X_train)) / np.full(X_test.shape, std_X_train)

    # useful for scaling additive noise
    if normalize_labels:
        mean_y_train = np.mean(y_train)
        std_y_train = np.std(y_train)
    else:
        mean_y_train = np.zeros(y_train.shape)
        std_y_train = np.ones(y_train.shape)

    y_train = (y_train - mean_y_train) / std_y_train
    y_test = (y_test - mean_y_train) / std_y_train

    # Printing the size of the training, validation and test sets
    print('Number of training examples: ' + str(X_train.shape[0]))
    print('Number of test examples: ' + str(X_test.shape[0]))

    sample_results = {"sotl": [], "naive_sotl": [], "gen": []}
    for d_prob in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        print("Dropout: {}".format(d_prob))
        sotl, naive_sotl, gen = train.sample_then_optimize(d_prob, X_train, y_train, X_test, y_test)
        sample_results["sotl"].append(sotl)
        sample_results["naive_sotl"].append(naive_sotl)
        sample_results["gen"].append(gen)

    print("Sample SoTL:")
    print(sample_results["sotl"])
    print("Naive SoTL:")
    print(sample_results["naive_sotl"])
    print("Gen:")
    print(sample_results["gen"])

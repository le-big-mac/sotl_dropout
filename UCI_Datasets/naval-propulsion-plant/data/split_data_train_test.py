from sklearn.model_selection import KFold
import numpy as np

# We set the random seed

np.random.seed(1)

# We load the data

data = np.loadtxt('data.txt')
n = data.shape[ 0 ]

# We generate the training test splits
indices_to_use = np.random.choice(range(n), 1000, replace = False)

n_splits = 20
kf = KFold(n_splits=20)
for i, (train, test) in enumerate(kf.split(range(1000))):
    index_train = indices_to_use[train]
    index_test = indices_to_use[test]

    np.savetxt("index_train_{}.txt".format(i), index_train, fmt = '%d')
    np.savetxt("index_test_{}.txt".format(i), index_test, fmt = '%d')

np.savetxt("n_splits.txt", np.array([ n_splits ]), fmt = '%d')

# We store the index to the features and to the target

index_features = np.array(range(data.shape[ 1 ] - 2), dtype = int)
index_target = np.array([ data.shape[ 1 ] - 2 ])

np.savetxt("index_features.txt", index_features, fmt = '%d')
np.savetxt("index_target.txt", index_target, fmt = '%d')

# We store the number of hidden neurons to use

np.savetxt("n_hidden.txt", np.array([ 50 ]), fmt = '%d')

# We store the number of epochs to use

np.savetxt("n_epochs.txt", np.array([ 40 ]), fmt = '%d')

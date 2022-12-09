# Imports
import numpy as np
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import torch
from progress.bar import Bar

import random


# Load training set
# List containing propbank info:
#   inst level:
#       sent (list), sent id (int), fileid (str), arg (list):
#           arg contents (list), arg length (int), arg index (tuple), role (ARG1/ARG0), hidden states for range
training_set = pickle.load(open("training_set_multi.p", "rb"))

print(len(training_set))
# Load classifier, using scikit learn's defaults
model = SGDClassifier()

# Get activation for each layer for each test item (the garden paths)

# Layers to test on
#layers = [0, 5, 11]
layer = 8

# 0.1, 1, 10, 100% of the training data
#training_set_sizes = [50, 500, 5000]
#50000]
training_set_sizes = [5000]

#weight_decay_params = [0.01, 1, 1.0, 10.0]
weight_decay_params = [0.01]

param_sets = [(x, y) for x in training_set_sizes for y in weight_decay_params]

results = [["sample_size", "weight_decay", "arg0_score", "arg0_selec", "arg1_score", "arg1_selec", "control_score", "arg0_samples", "arg1_samples", "arg0_arg1_crossover"]]

#print(param_sets)


for param in param_sets:

    train_size = param[0]
    train_size_half = train_size//2
    print(train_size)
    print(train_size_half)
    weight_decay = param[1]

    # Initialize training lists
    vectors = []
    arg0_labels = []
    arg1_labels = []
    control_labels = []

    layer_index = torch.LongTensor(range(layer, layer+1))
    
    # loop through and subselect for layer
    for word in training_set:
        vector = word[0]
        vectors.append(vector)
        arg0_labels.append(word[1])
        arg1_labels.append(word[2])
        control_labels.append(word[3])
        
    # Get numbers of training items for each set
    arg0_samples = train_size_half
    arg1_samples = train_size_half

    #---------------- Generate randomized, length matched training sets ----------------#

    # Get indexes of arg1s
    arg0_all_1_idxes = [idx for idx, x in enumerate(arg0_labels) if x == 1]
    arg0_all_0_idxes = [idx for idx, x in enumerate(arg0_labels) if x == 0]
    arg1_all_1_idxes = [idx for idx, x in enumerate(arg1_labels) if x == 1]
    arg1_all_0_idxes = [idx for idx, x in enumerate(arg1_labels) if x == 0]
    control_all_1_idxes = [idx for idx, x in enumerate(control_labels) if x == 1]
    control_all_0_idxes = [idx for idx, x in enumerate(control_labels) if x == 1]

    # Shuffle order
    random.shuffle(arg0_all_1_idxes)
    random.shuffle(arg0_all_0_idxes)
    random.shuffle(arg1_all_1_idxes)
    random.shuffle(arg1_all_0_idxes)
    random.shuffle(control_all_1_idxes)
    random.shuffle(control_all_0_idxes)


    # Subselect to training size
    arg0_all_1_idxes = arg0_all_1_idxes[:train_size_half]
    arg0_all_0_idxes = arg0_all_0_idxes[:train_size_half]
    arg1_all_1_idxes = arg0_all_1_idxes[:train_size_half]
    arg1_all_0_idxes = arg0_all_0_idxes[:train_size_half]


    # Generate lists of vectors for all four
    arg0_1_vectors = [x for idx, x in enumerate(vectors) if idx in arg0_all_1_idxes]
    arg0_0_vectors = [x for idx, x in enumerate(vectors) if idx in arg0_all_0_idxes]
    arg1_1_vectors = [x for idx, x in enumerate(vectors) if idx in arg1_all_1_idxes]
    arg1_0_vectors = [x for idx, x in enumerate(vectors) if idx in arg1_all_0_idxes]

    # Generate control lables for Arg0, Arg1 classifeirs
    arg0_control_labels = [x for idx, x in enumerate(control_labels) if idx in arg0_all_1_idxes or idx in arg0_all_0_idxes]
    arg1_control_labels = [x for idx, x in enumerate(control_labels) if idx in arg1_all_1_idxes or idx in arg1_all_0_idxes]

    # Concatinate vectors
    print(arg0_1_vectors[0].shape)
    arg0_vectors = np.squeeze(np.concatenate((arg0_1_vectors, arg0_0_vectors), axis=0))
    arg1_vectors = np.squeeze(np.concatenate((arg1_1_vectors, arg1_0_vectors), axis=0))
    print(arg0_vectors.shape)

    # Make arrays of the labels
    arg0_labels = np.concatenate((np.array(np.ones(train_size_half)), np.array(np.zeros(train_size_half))))
    arg1_labels = np.concatenate((np.array(np.ones(train_size_half)), np.array(np.zeros(train_size_half))))
    arg0_control_labels = np.array(arg0_control_labels)
    arg1_control_labels = np.array(arg1_control_labels)


    #---------------- Arg0 Classifier ----------------#
    print([x.shape for x in [arg0_vectors, arg0_labels, arg0_control_labels]])
    # Train test split
    arg0_vectors_train, arg0_vectors_test, arg0_labels_train, arg0_labels_test, arg0_control_labels_train, arg0_control_labels_test = train_test_split(arg0_vectors, arg0_labels, arg0_control_labels)

    # Make pipeline
    arg0_clf = make_pipeline(SGDClassifier(loss="log_loss", penalty='l2', alpha=weight_decay, max_iter=1000))
    arg0_control_clf = make_pipeline(SGDClassifier(loss="log_loss", penalty='l2', alpha=weight_decay, max_iter=1000))

    # Train main and control classifiers
    arg0_clf.fit(arg0_vectors_train, arg0_labels_train)
    arg0_control_clf.fit(arg0_vectors_train, arg0_control_labels_train)

    # Score arg0 classifiers
    arg0_score = arg0_clf.score(arg0_vectors_test, arg0_labels_test)
    arg0_control_score = arg0_clf.score(arg0_vectors_test, arg0_control_labels_test)
    arg0_selec = arg0_score - arg0_control_score

    #---------------- Arg1 Classifier ----------------#
    # Train test split
    arg1_vectors_train, arg1_vectors_test, arg1_labels_train, arg1_labels_test, arg1_control_labels_train, arg1_control_labels_test = train_test_split(arg1_vectors, arg1_labels, arg1_control_labels)

    # Make pipeline
    arg1_clf = make_pipeline(SGDClassifier(loss="log_loss", penalty='l2', alpha=weight_decay, max_iter=1000))
    arg1_control_clf = make_pipeline(SGDClassifier(loss="log_loss", penalty='l2', alpha=weight_decay, max_iter=1000))

    # Train main and control classifiers
    arg1_clf.fit(arg1_vectors_train, arg1_labels_train)
    arg1_control_clf.fit(arg1_vectors_train, arg1_control_labels_train)

    # Score arg1 classifiers
    arg1_score = arg1_clf.score(arg1_vectors_test, arg1_labels_test)
    arg1_control_score = arg1_clf.score(arg1_vectors_test, arg1_control_labels_test)
    arg1_selec = arg1_score - arg1_control_score

    # "sample_size", "weight_decay", "arg0_score", "arg0_selec", "arg1_score", "arg1_selec", "control_score", "arg0_samples", "arg1_samples", "arg0_arg1_crossover"

    results.append([train_size, weight_decay, arg0_score, arg0_selec, arg1_score, arg1_selec, arg0_samples, arg1_samples])
    print([train_size, weight_decay, arg0_score, arg0_selec, arg1_score, arg1_selec, arg0_samples, arg1_samples])
    #pickle.dump(arg0_clf, open("arg0_clf.p", "wb"))
    #pickle.dump(arg1_clf, open("arg1_clf.p", "wb"))

print(results)
with open("hyperparameter_results.tsv", "w") as f:
    for x in results:
        f.write("\t".join([str(y) for y in x]))
        f.write("\n")

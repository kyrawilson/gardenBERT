# Imports
import numpy as np
from transformers import BertConfig
import pickle
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import torch
from progress.bar import Bar
import sqlite3

# Set hyperparameters + info vars
model_name = 'bert-base-uncased'
layer_count = BertConfig.from_pretrained(model_name).num_hidden_layers
training_set = 'training.squlite'
train_size = 10

# Load classifier, using scikit learn's defaults
model = SGDClassifier()
weight_decay = 0.01
model = SGDClassifier()

# Header for results file
results = ["layer", "sample_size", "weight_decay", "arg0_score", "arg0_selec", "arg1_score", "arg1_selec", "control_score"]

# Define sql connection 
conn = sqlite3.connect('training.sqlite')
cur = conn.cursor()

# sqlite retreive the training sets
def get_training(cursor, layer_index, training_size, label):
    
    train_size_half = training_size//2
    query_1 = f"SELECT {label}_tag, control_tag, layer{layer_index} FROM embeddings WHERE {label}_tag = 1 ORDER BY RANDOM() LIMIT {train_size_half}"
    query_0 = f"SELECT {label}_tag, control_tag, layer{layer_index} FROM embeddings WHERE {label}_tag = 0 ORDER BY RANDOM() LIMIT {train_size_half}"

    cursor.execute(query_1)
    data_1 = cursor.fetchall()

    cursor.execute(query_0)
    data_0 = cursor.fetchall()

    all_data = np.concatenate([pickle.loads(x[2]) for x in data_1] + [pickle.loads(x[2]) for x in data_0], axis=0)
    all_tags = np.array([1 for x in range(train_size_half)] + [0 for x in range(train_size_half)])
    all_control = np.array([x[1] for x in data_1] + [x[1] for x in data_0])

    return all_data, all_tags, all_control

# Train classifiers
for layer_index in range(layer_count):
    print(f"Processing layer: {layer_index}")

    arg0_vectors, arg0_labels, arg0_control_labels = get_training(cur, layer_index, train_size, 'arg0')
    arg1_vectors, arg1_labels, arg1_control_labels = get_training(cur, layer_index, train_size, 'arg1')

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

    results.append([layer_index, train_size, weight_decay, arg0_score, arg0_selec, arg1_score, arg1_selec])
    print([layer_index, train_size, weight_decay, arg0_score, arg0_selec, arg1_score, arg1_selec])
    #pickle.dump(arg0_clf, open("arg0_clf.p", "wb"))
    #pickle.dump(arg1_clf, open("arg1_clf.p", "wb"))

print(results)

with open("hyperparameter_results.tsv", "w") as f:
    for x in results:
        f.write("\t".join([str(y) for y in x]))
        f.write("\n")

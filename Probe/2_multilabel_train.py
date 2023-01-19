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

#---------------- Setup ----------------#

# Set hyperparameters + info vars
training_set = 'training.squlite'
train_size = 10

# Load classifier, using scikit learn's defaults
model = SGDClassifier()
weight_decay = 0.01
model = SGDClassifier()

# Header for results file
results = ["model_name", "layer", "sample_size", "weight_decay", "arg0_score", "arg0_selec", "arg1_score", "arg1_selec", "control_score"]

# Models being tested
models = ['bert-large-uncased', 'bert-base-uncased', 'bert-base-cased', 'roberta-base', 'roberta-large']

# Define sql connection 
conn = sqlite3.connect('training.sqlite')
cur = conn.cursor()

    
# sqlite retreive the training sets
def get_training(cursor, model_name, layer_index, training_size, label):
    
    train_size_half = training_size//2

    # Get data from database
    query = f"SELECT arg{label}_tag, control_tag, layer{layer_index} FROM {model_name.replace('-', '_')} WHERE classifier == {label}"
    cursor.execute(query)
    data = cursor.fetchall()

    # Convert to numpy arrays
    all_data = np.squeeze(np.array([pickle.loads(x[2]) for x in data]))
    all_tags = np.array([x[0] for x in data])
    all_control = np.array([x[1] for x in data])

    return all_data, all_tags, all_control

# Train classifiers

for model in models:

    layer_count = BertConfig.from_pretrained(model).num_hidden_layers

    for layer_index in range(layer_count):
        print(f"Processing layer {layer_index} of model {model}")

        arg0_vectors, arg0_labels, arg0_control_labels = get_training(cur, model, layer_index, train_size, '0')
        arg1_vectors, arg1_labels, arg1_control_labels = get_training(cur, model, layer_index, train_size, '1')

        print(f"arg0_vector size is {arg0_vectors.shape}, arg0_labels size is {arg0_labels.shape}, control_labels size is {arg0_control_labels.shape}")
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

        results.append([model, layer_index, train_size, weight_decay, arg0_score, arg0_selec, arg1_score, arg1_selec])
        print([layer_index, train_size, weight_decay, arg0_score, arg0_selec, arg1_score, arg1_selec])
        #pickle.dump(arg0_clf, open("arg0_clf.p", "wb"))
        #pickle.dump(arg1_clf, open("arg1_clf.p", "wb"))

print(results)

with open("hyperparameter_results.tsv", "w") as f:
    for x in results:
        f.write("\t".join([str(y) for y in x]))
        f.write("\n")

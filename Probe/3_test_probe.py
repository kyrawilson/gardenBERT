# Imports
from cgi import test
import torch
import pickle
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd 
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

# BERT, Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )


# Layer to probe
layer = 8

# Find sublist function, ripped shamelessly from stackexchange
def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1


# Read in sentences, get uique and First two slots of code
gps = pd.read_csv("garden_paths.tsv", sep="\t", header=0)

# activation, arg0, arg1, n1/n2, sent type
test_set = []

for index, row in gps.iterrows():

    # Tokenize and index sentence and arguments for input
    sentence_tok = tokenizer.tokenize("[CLS] " + row['context'] + " [SEP]")
    sentence_idxed = tokenizer.convert_tokens_to_ids(sentence_tok)
    print(sentence_tok)

    N1_tok = tokenizer.tokenize(row['p1'])
    print(N1_tok)
    N1_idxed = tokenizer.convert_tokens_to_ids(N1_tok)
    N2_tok = tokenizer.tokenize(row['p2'])
    print(N2_tok)
    N2_idxed = tokenizer.convert_tokens_to_ids(N2_tok)
    
    N1_loc = find_sub_list(N1_idxed, sentence_idxed)
    N2_loc = find_sub_list(N2_idxed, sentence_idxed)

    # Tag words for acc testing
    n1_tags = []
    n2_tags = []

    if row['code'][:2] == 'LC':
        # Arg0, Arg1
        n1_tags = [1, 0]
        n2_tags = [0, 1]

    else:
        # Arg0, Arg1
        n1_tags = [1, 1]
        n2_tags = [0, 1]

    # Get BERT activation
    segments_tensor = torch.tensor([[1] * len(sentence_idxed)])
    tokens_tensor = torch.tensor([sentence_idxed])
    outputs = model(tokens_tensor, segments_tensor)
    hidden_states = torch.stack(outputs[2])

    # 8th layer selection tensor
    layer_index = torch.LongTensor(range(layer, layer+1))

    if len(N1_idxed) > 1:
        
        N1_act = hidden_states.index_select(0, layer_index).detach().numpy().squeeze()
        N1_act = np.mean(N1_act[N1_loc[0]:N1_loc[1] + 1, :], axis=0)
        #print(N1_act.shape)
    else:
        N1_act = hidden_states.index_select(0, layer_index).detach().numpy().squeeze()[N1_loc[0]:N1_loc[1] + 1].squeeze()
        #print(N1_act.shape)
    
    if len(N2_idxed) > 1:
        
        N2_act = hidden_states.index_select(0, layer_index).detach().numpy().squeeze()
        N2_act = np.mean(N2_act[N2_loc[0]:N2_loc[1] + 1, :], axis=0)
        #print(N2_act.shape)

    else:
        N2_act = hidden_states.index_select(0, layer_index).detach().numpy().squeeze()[N2_loc[0]:N2_loc[1] + 1].squeeze()
        #print(N2_act.shape)
    
    if N1_act.shape != (768,):
        print('Vector embedding is incorrect dimensions')
    if N2_act.shape != (768,):
        print("Vector embedding is incorrect dimensions")
    # Append to testing set
    # activation, n1_vec, n1_arg0, n1_arg1, n2_vec, n2_arg0, n2_arg1, code, sentence
    test_set.append([row['context'], N1_act, n1_tags[0], n1_tags[1], N2_act, n2_tags[0], n2_tags[1]])

act_df = pd.DataFrame(test_set, columns=['context', 'n1_vec', 'n1_arg0', 'n1_arg1', 'n2_vec', 'n2_arg0', 'n2_arg1'])
# Test classifier

final_df = gps.merge(act_df, on='context', how='outer')

#print(final_df.iloc[40])
#print(f"shape of A is {act_df.shape}")
#print(f"shape of B is {gps.shape}")
#print(f"shape of pd.concat([A, B], axis=1) is {final_df.shape}")

#print([final_df.iloc[idx] for idx, x in enumerate(n1_just_vectors) if type(x) == float])
#print([final_df.iloc[idx] for idx, x in enumerate(n2_just_vectors) if type(x) == float])

# Import Classifier
arg0_clf = pickle.load(open("arg0_clf.p", "rb"))
arg1_clf = pickle.load(open("arg1_clf.p", "rb"))

#Classifier results
columns = ['context', "n1_arg0_label", "n1_arg0_0_prob", "n1_arg0_1_prob", "n1_arg1_label", "n1_arg1_0_prob", "n1_arg1_1_prob", "n2_arg0_label", "n2_arg0_0_prob", "n2_arg0_1_prob", "n2_arg1_label", "n2_arg1_0_prob", "n2_arg1_1_prob"]
results = []
#print(final_df.iloc[35])
#print(act_df.iloc[35])

for index, row in final_df.iterrows():

    n1 = np.expand_dims(np.array(row['n1_vec']), axis=0)
    n2 = np.expand_dims(np.array(row['n2_vec']), axis=0)
    
    n1_arg0_prob = arg0_clf.predict_proba(n1)
    n1_arg0_label = arg0_clf.predict(n1)
    n1_arg0_0_prob = n1_arg0_prob[0][0]
    n1_arg0_1_prob = n1_arg0_prob[0][1]

    n1_arg1_prob = arg1_clf.predict_proba(n1)
    n1_arg1_label = arg1_clf.predict(n1)
    n1_arg1_0_prob = n1_arg1_prob[0][0]
    n1_arg1_1_prob = n1_arg1_prob[0][1]


    n2_arg0_probs = arg0_clf.predict_proba(n2)
    n2_arg0_label = arg0_clf.predict(n2)
    n2_arg0_0_prob = n2_arg0_probs[0][0]
    n2_arg0_1_prob = n2_arg0_probs[0][1]

    n2_arg1_probs = arg1_clf.predict_proba(n2)
    n2_arg1_label = arg1_clf.predict(n2)
    n2_arg1_0_prob = n2_arg1_probs[0][0]
    n2_arg1_1_prob = n2_arg1_probs[0][1]
    results.append([row['context'], n1_arg0_label, n1_arg0_0_prob, n1_arg0_1_prob, n1_arg1_label, n1_arg1_0_prob, n1_arg1_1_prob, n2_arg0_label, n2_arg0_0_prob, n2_arg0_1_prob, n2_arg1_label, n2_arg1_0_prob, n2_arg1_1_prob])


results_df = pd.DataFrame(results, columns=columns)
print(results_df.iloc[40]['context'])
print(type(results_df.iloc[40]['context']))

print(final_df.iloc[40]['context'])
print(type(final_df.iloc[40]['context']))

results_df['context'] = results_df['context'].astype(object)
final_df['context'] = final_df['context'].astype(object)

print(final_df.columns)
print(results_df.columns)

new_df = pd.concat([final_df, results_df], axis='columns')
new_df = new_df.drop(labels=['n1_vec', 'n2_vec'], axis='columns')

new_df.to_csv("probe_results.csv")


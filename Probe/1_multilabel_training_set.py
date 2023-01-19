
# Imports
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from nltk.corpus import propbank as pb
from nltk.corpus import treebank as tb
import pickle
from progress.bar import Bar
import random
import sqlite3
import pandas as pd
import os

#---------------- Setup ----------------#
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
models = ['bert-large-uncased', 'bert-base-uncased', 'bert-base-cased', 'roberta-base', 'roberta-large']

training_size = 10
training_size_half = training_size // 2
token_filename = 'pb_tokens.p'

conn = sqlite3.connect('training.sqlite')
cur = conn.cursor()

# Find sublist function (https://stackoverflow.com/questions/17870544/find-starting-and-ending-indices-of-sublist-in-list)
def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1

#---------------- Generating Training Tokens ----------------#

if os.path.isfile(token_filename):
    print(f'Loading file: {token_filename}')
    with open('pb_tokens.p', "rb") as f:
         pb_tokens = pickle.load(f)

else:
    # Load propbank using nltk
    pb_instances = pb.instances()[:1000]

    pb_tokens = pd.DataFrame(columns=['token_index', 'source_sent', 'arg0_tag', 'arg1_tag'])

    # List containing propbank info:
    #   inst level:
    #       sent (list), sent id (int), fileid (str), arg (list):
    #           arg contents (list), arg length (int), arg index (tuple), role (ARG1/ARG0)
    # Important items are ARG0 (agent) and ARG1 (patient)

    pb_sents = {}

    goalcount = 0
    agentcount = 0

    # Loop through propbank
    with Bar("Reading in propbank instances", max=len(pb_instances)) as bar:
        for inst in pb_instances:
            # variables of instance
            file = inst.fileid
            sent_num = inst.sentnum
            sent_id = f"{file}_{sent_num}"

            try:
                pb_sents[sent_id].append(inst)
            
            except KeyError:
                pb_sents[sent_id] = [inst]

            bar.next()

    training_set = []
    control_dict = {}

    keycount = 0

    training_sentences = []

    with Bar("Reading propbank tokens", max=len(pb_sents.keys())) as bar:
        # Loop through each unique sentence
        for key in pb_sents.keys():

            # Propbank instance info
            file = pb_sents[key][0].fileid
            sent_num = pb_sents[key][0].sentnum

            try:
                raw_sent = tb.sents(file)[sent_num]
            
            except OSError:
                print(f'file not found: {file}')
                continue

            tree = tb.sents(file)[sent_num]
            sent_tok = tokenizer.tokenize("[CLS] " + " ".join(tree) + " [SEP]")
            sent_idxed = tokenizer.convert_tokens_to_ids(sent_tok)

            # Tokenize sentence

            arg0_locs = []
            arg1_locs = []

            for inst in pb_sents[key]:
                # Get verb instance
                verb_frame = inst.roleset

                # Get verb's roles, read into dictionary to match with thematic tags
                roles = {}

                try:
                    for role in pb.roleset(verb_frame).findall("roles/role"):
                        roles[f'ARG{role.attrib["n"]}'] = role.attrib['f']
                except:
                    continue
                arg_tags = [x[1] for x in inst.arguments]

                # Check for ARGS in arg list of instance and assign location
                # ARG0 must have PAG tag (propbank proto agent)
                try:
                    if 'ARG0' in arg_tags and roles['ARG0'] == 'PAG':
                        # Get argument position from propbank tree pinter
                        arg0_pos = [x for x in inst.arguments if x[1] == 'ARG0'][0][0]

                        # Select tree leaves with argument
                        arg0 = arg0_pos.select(inst.tree).leaves()

                        # Tokenize using BERT's tokenizer
                        arg0_tok = tokenizer.tokenize(" ".join(arg0))

                        # Get the indecies for the argument
                        arg0_idx = find_sub_list(arg0_tok, sent_tok)

                        # Convert tokenized words to BERT's IDs
                        arg0_idxed = tokenizer.convert_tokens_to_ids(arg0_tok)

                        # Find those within the sentence
                        arg0_loc = find_sub_list(arg0_idxed, sent_idxed)

                        # Add their locations to the list
                        arg0_locs.append(arg0_loc)

                        # count number of training samples
                        agentcount += 1

                except KeyError:
                    print(f'\nverb {verb_frame} has no arg0')

                # Roles to generalize over for "Patient" Classifer
                ppt_roles = ['GOL']

                # Itterate through remaining arguments
                if len(arg_tags) > 1:
                    for tag in arg_tags[1:]:
                        
                        # Check if it's one of the correct patient-y roles
                        try:
                            if roles[tag] in ppt_roles:
                                arg1_pos = [x for x in inst.arguments if x[1] == tag][0][0]
                                arg1 = arg1_pos.select(inst.tree).leaves()
                                arg1_tok = tokenizer.tokenize(" ".join(arg1))
                                arg1_idx = find_sub_list(arg1_tok, sent_tok)
                                arg1_idxed = tokenizer.convert_tokens_to_ids(arg1_tok)
                                arg1_loc = find_sub_list(arg1_idxed, sent_idxed)
                                arg1_locs.append(arg1_loc)
                                goalcount += 1
                        except KeyError:
                            pass
            
            # Write list of indexed words, plus 1/0 for each word for both arg types
            arg0_tags = [0] * len(sent_idxed)
            arg1_tags = [0] * len(sent_idxed)
            control_tags = [0] * len(sent_idxed)

            for arg0 in arg0_locs:
                if arg0 != None:
                    for y in [x for x in range(arg0[0], arg0[1] + 1)]:
                        arg0_tags[y] = 1
                        
            for arg1 in arg1_locs:
                if arg1 != None:
                    for y in [x for x in range(arg1[0], arg1[1] + 1)]:
                        arg1_tags[y] = 1
            
            # Assign control tags for each word
            for idx, word in enumerate(sent_idxed):
                
                try:
                    control_tag = control_dict[sent_idxed[idx]]
                    control_tags[idx] = control_tag

                except KeyError:
                    control_tag = random.randint(0, 1)
                    control_dict[sent_idxed[idx]] = control_tag
                    control_tags[idx] = control_tag
            
            # Write token information to big df of all tokens
            for x in range(len(sent_idxed)):
                token_index = torch.LongTensor(range(x, x+1))
                arg0_tag = arg0_tags[x]
                arg1_tag = arg1_tags[x]
                control_tag = control_tags[x]

                pb_tokens.loc[len(pb_tokens)] = [token_index, sent_tok, arg0_tag, arg1_tag, control_tag]
            bar.next()
        

    with open('pb_tokens.p', "wb") as f:
        pickle.dump(pb_tokens, f)

# Set the training sets
arg0_1_training_tokens = pb_tokens[pb_tokens['arg0_tag'] == 1].sample(n=training_size_half)
arg0_0_training_tokens = pb_tokens[pb_tokens['arg0_tag'] == 0].sample(n=training_size_half)

arg0_1_training_tokens['classifier'] = 0
arg0_0_training_tokens['classifier'] = 0

arg1_1_training_tokens = pb_tokens[pb_tokens['arg1_tag'] == 1].sample(n=training_size_half)
arg1_0_training_tokens = pb_tokens[pb_tokens['arg1_tag'] == 0].sample(n=training_size_half)

arg1_1_training_tokens['classifier'] = 1
arg1_0_training_tokens['classifier'] = 1

all_training_tokens = pd.concat([arg0_1_training_tokens, arg0_0_training_tokens, arg1_1_training_tokens, arg1_0_training_tokens])

#---------------- Getting Token Embeddings ----------------#
for model_name in models:

    # Load BERT Model
    model = BertModel.from_pretrained(model_name,
                                    output_hidden_states = True, # Whether the model returns all hidden-states.
                                    )
    model.eval()

    # Get model info
    layer_count = BertConfig.from_pretrained(model_name).num_hidden_layers

    layer_info = {}

    for x in range(layer_count):
        layer_name = f"layer{x}"
        layer_index = torch.LongTensor(range(x, x+1))
        layer_info[x] = {'layer_name': layer_name, 'layer_index': layer_index}

    # Initialize sql table based on number of layers
    create_query = """CREATE TABLE """ + model_name.replace('-', '_') + """ (
        inst_id INTEGER PRIMARY KEY,
        arg0_tag INTEGER NOT NULL,
        arg1_tag INTEGER NOT NULL,
        control_tag INTEGER NOT NULL,
        classifier INTEGER NOT NULL,
        """ + ',\n'.join([f'layer{x} BLOB NOT NULL' for x in layer_info.keys()]) + ');'

    cur.execute(create_query)

    # Itterate through training samples
    keycount = 0
    for index, row in all_training_tokens.iterrows():
        # Get BERT activation for sentence
        sent_idxed = tokenizer.convert_tokens_to_ids(row['source_sent'])
        #print(f'tokenized sentence is: {row["source_sent"]}')
        #print(f"indexed sentence is: {sent_idxed}")

        segments_tensor = torch.tensor([[1] * len(sent_idxed)])
        tokens_tensor = torch.tensor([sent_idxed])
        try:
            outputs = model(tokens_tensor, segments_tensor)
        except:
            print(f'Skipped sentence: {row["source_sent"]}')
            continue

        hidden_states = torch.stack(outputs[2])

        # Write word layers, arg0 1/0, arg1 1/0 as list to training sent
        word_loc = row['token_index']

        # presevere dim 0 and 3, layer and hidden states respectively
        word_states = hidden_states.index_select(2, word_loc)
        word_states = torch.squeeze(word_states)
        #word_states = word_states.detach().numpy()

        # Add them to a list
        layers = []

        for layer in layer_info.keys():
            layer_vec = word_states.index_select(0, layer_info[layer]['layer_index']).detach().numpy()
            layers.append(sqlite3.Binary(pickle.dumps(layer_vec, pickle.HIGHEST_PROTOCOL)))
        # word_states = word_states.index_select(0, layer_index).detach().numpy()

        query = f"""INSERT INTO """ + model_name.replace('-', '_') + f"""
                (inst_id, arg0_tag, arg1_tag, control_tag, classifier, {', '.join([layer_info[x]['layer_name'] for x in layer_info.keys()])})
                VALUES
                ({keycount}, {row['arg0_tag']}, {row['arg1_tag']}, {row['control_tag']}, {row['classifier']}, {', '.join(['?' for x in range(layer_count)])})"""
        
        cur.execute(query, tuple(layers))
        #training_set.append([word_states, arg0_tags[idx], arg1_tags[idx], control_tag])

        keycount += 1

conn.commit()
conn.close()
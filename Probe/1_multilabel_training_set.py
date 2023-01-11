
# Imports
import torch
from transformers import BertTokenizer, BertModel
from nltk.corpus import propbank as pb
from nltk.corpus import treebank as tb
import pickle
from progress.bar import Bar
import random
import sqlite3


# Find sublist function (https://stackoverflow.com/questions/17870544/find-starting-and-ending-indices-of-sublist-in-list)
def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1

# Load BERT Model and Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True, # Whether the model returns all hidden-states.
                                  )
model.eval()


# Load propbank using nltk
pb_instances = pb.instances()[:10000]

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

# Which layers to save
layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
layer = 8
layer_index = torch.LongTensor(range(layer, layer+1))


with Bar("Getting BERT activation", max=len(pb_sents.keys())) as bar:

    # Loop through each unique sentence
    for key in pb_sents.keys():

        # Propbank instance info
        file = pb_sents[key][0].fileid
        sent_num = pb_sents[key][0].sentnum
        raw_sent = tb.sents(file)[sent_num]
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

        for arg0 in arg0_locs:
            if arg0 != None:
                for y in [x for x in range(arg0[0], arg0[1] + 1)]:
                    arg0_tags[y] = 1
                    
        for arg1 in arg1_locs:
            if arg1 != None:
                for y in [x for x in range(arg1[0], arg1[1] + 1)]:
                    arg1_tags[y] = 1

        # Get BERT activation for sentence
        segments_tensor = torch.tensor([[1] * len(sent_idxed)])
        tokens_tensor = torch.tensor([sent_idxed])
        outputs = model(tokens_tensor, segments_tensor)
        hidden_states = torch.stack(outputs[2])

        # Write word layers, arg0 1/0, arg1 1/0 as list to training sent
        for idx, word in enumerate(sent_idxed):
            
            try:
                control_tag = control_dict[sent_idxed[idx]]

            except KeyError:
                control_tag = random.randint(0, 1)
                control_dict[sent_idxed[idx]] = control_tag


            word_loc = torch.LongTensor(range(idx, idx+1))

            # presevere dim 0 and 3, layer and hidden states respectively
            word_states = hidden_states.index_select(2, word_loc)
            word_states = torch.squeeze(word_states)
            word_states = word_states.detach().numpy()

            # word_states = word_states.index_select(0, layer_index).detach().numpy()

            training_set.append([word_states, arg0_tags[idx], arg1_tags[idx], control_tag])
        bar.next()

print(f'number of -GOL training instances is {goalcount}')
print(f'number of -GOL training instances is {agentcount}')
pickle.dump(training_set, open("output/training_set_multi.p", "wb"))


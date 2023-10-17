import json
from copy import deepcopy

train_file_path = 'train.json'
valid_file_path = 'valid.json'
context_file_path = 'context.json'
swag_file_path = 'train_swag.json'

def read_json(path):
    with open(path, 'r') as open_context:
        file = json.load(open_context)
    
    return file
def write_json(path, content):
    with open(path, 'w') as write_context:
        json.dump(content, write_context)
        
    return

train_file = read_json(train_file_path)
context_file = read_json(context_file_path)

swag_all = []

swag_template_ = {
    "video-id": "", # train ID
    "fold-ind": "", # ignore
    "startphrase": "", # ignore
    "sent1": "", # question
    "sent2": "", # ignore
    "gold-source": "", # ignore
    "ending0": "",
    "ending1": "",
    "ending2": "",
    "ending3": "",
    "label":"" # relevent
}

for cnt in range(0,len(train_file)):
    swag_template = deepcopy(swag_template_)
    swag_template["video-id"] = train_file[cnt]["id"]
    swag_template["sent1"] = train_file[cnt]["question"]
    swag_template["ending0"] = context_file[train_file[cnt]["paragraphs"][0]]
    swag_template["ending1"] = context_file[train_file[cnt]["paragraphs"][1]]
    swag_template["ending2"] = context_file[train_file[cnt]["paragraphs"][2]]
    swag_template["ending3"] = context_file[train_file[cnt]["paragraphs"][3]]
    for para_cnt in range(0,4):
        if(train_file[cnt]["paragraphs"][para_cnt] == train_file[cnt]["relevant"]):
            swag_template["label"] = para_cnt
        else:
            continue
    swag_all.append(swag_template)

write_json(swag_file_path, swag_all)    
import json
from copy import deepcopy

train_file_path = 'train.json'
valid_file_path = 'valid.json'
train_squad_file_path = 'train_squad.json'
valid_squad_file_path = 'valid_squad.json'
context_file_path = 'context.json'

def read_json(path):
    with open(path, 'r') as open_context:
        file = json.load(open_context)
    
    return file
def write_json(path, content):
    with open(path, 'w') as write_context:
        json.dump(content, write_context)
        
    return

train_file = read_json(train_file_path)
valid_file = read_json(valid_file_path)
context_file = read_json(context_file_path)

train_squad_all = []
valid_squad_all = []

squad_template_ = {
    "id":"",
    "title":"",
    "context":"",
    "question":"",
    "answers": {"text":[],"answer_start":[]}
}

for cnt in range(0,len(train_file)):
    squad_template = deepcopy(squad_template_)
    squad_template["id"] = train_file[cnt]["id"]
    squad_template["context"] = context_file[train_file[cnt]["relevant"]]
    squad_template["question"] = train_file[cnt]["question"]
    squad_template["answers"]["text"].append(train_file[cnt]["answer"]["text"])
    squad_template["answers"]["answer_start"].append(train_file[cnt]["answer"]["start"])
    
    train_squad_all.append(squad_template)
    
write_json(train_squad_file_path,train_squad_all)

for cnt in range(0,len(valid_file)):
    squad_template = deepcopy(squad_template_)
    squad_template["id"] = valid_file[cnt]["id"]
    squad_template["context"] = context_file[valid_file[cnt]["relevant"]]
    squad_template["question"] = valid_file[cnt]["question"]
    squad_template["answers"]["text"].append(valid_file[cnt]["answer"]["text"])
    squad_template["answers"]["answer_start"].append(valid_file[cnt]["answer"]["start"])
    
    train_squad_all.append(squad_template)
    
write_json(valid_squad_file_path,valid_squad_all)
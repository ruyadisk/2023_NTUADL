import json

train_file_path = 'train.json'
valid_file_path = 'valid.json'
context_file_path = 'context.json'
train_squad_file_path = 'train_squad.json'
valid_squad_file_path = 'valid_squad.json'

def read_json(path):
    with open(path, 'r') as open_context:
        file = json.load(open_context)
    
    return file


train_swag_json = read_json(train_squad_file_path)
train_json = read_json(train_file_path)
valid_swag_json = read_json(valid_squad_file_path)
valid_json = read_json(valid_file_path)

print("TRAIN SIZE COMPARE: ", len(train_swag_json), len(train_json))
print("VALID SIZE COMPARE: ", len(valid_swag_json), len(valid_json))
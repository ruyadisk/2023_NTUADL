import argparse
import json
from dataclasses import dataclass
from itertools import chain
import pandas 
import torch
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from huggingface_hub import Repository, create_repo
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    pipeline,
)
from transformers.utils import PaddingStrategy, check_min_version, send_example_telemetry

def parse_args():
    parser = argparse.ArgumentParser(description="inference parsing")
    parser.add_argument("--context_file",type=str)
    parser.add_argument("--test_json_file",type=str)
    parser.add_argument("--output_csv_file",type=str)
    parser.add_argument("--max_seq_len",type=int,default=512)
    parser.add_argument("--per_device_eval_batch_size",type=int,default=64)
    
    args = parser.parse_args()
    return args

def read_json(path):
    with open(path, 'r') as open_context:
        file = json.load(open_context)
    return file

args = parse_args()
context_file = read_json(args.context_file)
padding = "max_length"

paragraph_selection_tokenizer = AutoTokenizer.from_pretrained("./Paragraph_Selection_pdtbs_4_gas_32_epochs_3")
paragraph_selection_model = AutoModelForMultipleChoice.from_pretrained("./Paragraph_Selection_pdtbs_4_gas_32_epochs_3")


def preprocess_function(examples):
    first_sentences = [[question] * 4 for question in examples["question"]] #quest
    second_sentences = [
        (context_file[index] for index in indexs) for indexs in examples["paragraphs"]
    ]

    # Flatten out
    first_sentences = list(chain(*first_sentences))
    second_sentences = list(chain(*second_sentences))

    # Tokenize
    tokenized_examples = paragraph_selection_tokenizer(
        first_sentences,
        second_sentences,
        max_length=args.max_seq_len,
        padding=padding,
        truncation=True,
    )
    # Un-flatten
    tokenized_inputs = {k: [v[i : i + 4] for i in range(0, len(v), 4)] for k, v in tokenized_examples.items()}
    return tokenized_inputs

data_files = {}
data_files["test"] = args.test_json_file
extension = data_files["test"].split(".")[-1]
paragraph_selection_test_datasets = load_dataset(extension, data_files=data_files)
processed_datasets = paragraph_selection_test_datasets.map(
    preprocess_function, batched=True, remove_columns=paragraph_selection_test_datasets["test"].column_names
)
print(paragraph_selection_test_datasets["test"]["id"])
eval_dataset = processed_datasets["test"]
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size)

paragraph_selection_model.to('cuda')
paragraph_selection_model.eval()
paragraph_selection_model_result = []
for batch in tqdm(eval_dataloader):
    with torch.no_grad():
        batch = {k: v.to('cuda') for k, v in batch.items()}
        outputs = paragraph_selection_model(**batch)
    predictions = outputs.logits.argmax(dim=-1)
    paragraph_selection_model_result.append(predictions)
    
paragraph_selection_model_result = list(chain(*paragraph_selection_model_result))

question = paragraph_selection_test_datasets["test"]["question"]
context = [context_file[paragraph_selection_test_datasets["test"]["paragraphs"][i][j.item()]] for i, j in enumerate(paragraph_selection_model_result)]
print("Length check!! ",len(question), " ", len(context))
span_selection_pipeline = pipeline("question-answering", model="./Span_Selection_pdtbs_2_gas_32_epochs_3",device=0)
res_ans = span_selection_pipeline(question=question, context=context)
ans = []
for it in res_ans:
    ans.append(it["answer"])

res_id = paragraph_selection_test_datasets["test"]["id"]

res = pandas.DataFrame(data={'id':res_id,"answer":ans})
res.to_csv(args.output_csv_file,index=False)

import argparse
from copy import deepcopy
from random import Random, random
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from utils import get_bnb_config, get_prompt
import json
from tqdm import tqdm
import copy
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="",
        help="Path to the checkpoint of Taiwan-LLM-7B-v2.0-chat. If not set, this script will use "
        "the checkpoint from Huggingface (revision = 5073b2bbc1aa5519acdc865e99832857ef47f7c9)."
    )
    parser.add_argument(
        "--peft_path",
        type=str,
        required=True,
        help="Path to the saved PEFT checkpoint."
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="",
        required=True,
        help="Path to test data."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="",
        required=True,
        help="Path to output data."
    )
    args = parser.parse_args()
    
    base_model_path = args.base_model_path
    peft_model_path = args.peft_path
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    model = AutoModelForCausalLM.from_pretrained(base_model_path, quantization_config=get_bnb_config(), load_in_4bit=True, device_map="auto")
    model = PeftModel.from_pretrained(model, peft_model_path)
    
    model.eval()
    test_file_path = args.test_data_path
    with open(test_file_path, "r") as f:
        test_file = json.load(f)
    
    progressbar = tqdm(range(len(test_file)))
     
    outputs = []
    outputs_sample = {"id":"","output":""} 
    
    split_symbol = "助理："
    
    for each_statement in test_file:
        inputs = tokenizer(get_prompt(each_statement["instruction"]), return_tensors="pt").to(device)
        tmp = copy.deepcopy(outputs_sample)
        tmp["id"] = each_statement["id"]
        tmp["output"] = (tokenizer.decode(model.generate(**inputs, max_new_tokens=256)[0], skip_special_tokens=True))
        index = tmp["output"].find(split_symbol)
        tmp["output"] = tmp["output"][index + len(split_symbol):].strip()
        outputs.append(tmp)
        progressbar.update(1)

    json_data = json.dumps(outputs, indent=2)
    with open(args.output_path, 'w') as file:
        file.write(json_data)
    
if __name__ == "__main__":
    main()
    
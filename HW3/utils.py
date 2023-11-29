from transformers import BitsAndBytesConfig
import torch

def get_prompt_fewshot(instruction: str, samples: list):
    fewshot_instruction_and_answer = f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。\n以下為幾個範例：\n"
    for each_sample in samples:
        fewshot_instruction_and_answer = fewshot_instruction_and_answer + f"用戶：{each_sample['instruction']} 助理：{each_sample['output']}\n"
    return fewshot_instruction_and_answer + f"請根據以上的範例回答下面的問題。\n用戶：{instruction} 助理："
        
def get_prompt(instruction: str) -> str:
    '''Format the instruction as a prompt for LLM.'''
    return f"你是人工智慧助理，以下是用戶和人工智能助理之間的對話。你要對用戶的問題提供有用、安全、詳細和禮貌的回答。\n請回答下面的問題。\n用戶: {instruction} 助理："

def get_bnb_config() -> BitsAndBytesConfig:
    '''Get the BitsAndBytesConfig.'''
    return BitsAndBytesConfig(
            load_in_4bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float32,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    pass

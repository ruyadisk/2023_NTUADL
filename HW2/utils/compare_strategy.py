from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("./summarize_pdtbs_2_pdebs_2_lr_5e5_gas_16_beam_30_epoch_10_cosine")
model = AutoModelForSeq2SeqLM.from_pretrained("./summarize_pdtbs_2_pdebs_2_lr_5e5_gas_16_beam_30_epoch_10_cosine")

input_ref = pd.read_json("data/public.jsonl", lines=True)
output_summaries_refs = []
output_summaries_greedy = []
output_summaries_beams_10 = []
output_summaries_top_k_30 = []
output_summaries_top_p_08 = []
output_summaries_temperature_06 = []

input_maintexts = input_ref["maintext"]
input_titles_ = input_ref["title"]
input_titles = []
for i in input_titles_:
    input_titles.append(i)
model.to("cuda:0")

progress_bar = tqdm(range(len(input_maintexts)))

for each_text in input_maintexts:
    each_id = tokenizer.encode(("summarize: " + each_text), return_tensors="pt", max_length=256, truncation=True).to("cuda")
    
    try:
        output_summaries_greedy.append( tokenizer.decode(model.generate(each_id, max_length=32, num_beams=1, no_repeat_ngram_size=5)[0], skip_special_tokens=True))
        output_summaries_beams_10.append( tokenizer.decode(model.generate(each_id, max_length=32, num_beams=20, no_repeat_ngram_size=5)[0], skip_special_tokens=True) )
        output_summaries_top_k_30.append( tokenizer.decode(model.generate(each_id, max_length=32, do_sample=True, top_k=15)[0], skip_special_tokens=True))
        output_summaries_top_p_08.append( tokenizer.decode(model.generate(each_id, max_length=32, do_sample=True, top_p=0.4)[0], skip_special_tokens=True))
        output_summaries_temperature_06.append( tokenizer.decode(model.generate(each_id, max_length=32, do_sample=True, temperature=0.3)[0], skip_special_tokens=True))
        progress_bar.update(1)
    except Exception as e:
        print(f"Exception occurred for text: {each_text}")
        print(f"Exception details: {str(e)}")

from tw_rouge import get_rouge
tw_rouge_score_greedy = get_rouge(output_summaries_greedy, input_titles)
tw_rouge_score_beams_10 = get_rouge(output_summaries_beams_10, input_titles)
tw_rouge_score_top_k_30 = get_rouge(output_summaries_top_k_30, input_titles)
tw_rouge_score_top_p_08 = get_rouge(output_summaries_top_p_08, input_titles)
tw_rouge_score_temperature_06 = get_rouge(output_summaries_temperature_06, input_titles)        

print(f"Greedy: {tw_rouge_score_greedy}")
print(f"Beams: {tw_rouge_score_beams_10}")
print(f"Top_k: {tw_rouge_score_top_k_30}")
print(f"Top_p: {tw_rouge_score_top_p_08}")
print(f"Temperature: {tw_rouge_score_temperature_06}")
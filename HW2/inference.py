from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import nltk
from tqdm import tqdm
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default=None,
    ),
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print(args.input_jsonl)
    tokenizer = AutoTokenizer.from_pretrained("./summarize_pdtbs_2_pdebs_2_lr_5e5_gas_16_beam_30_epoch_10_cosine")
    model = AutoModelForSeq2SeqLM.from_pretrained("./summarize_pdtbs_2_pdebs_2_lr_5e5_gas_16_beam_30_epoch_10_cosine")

    input_ref = pd.read_json(args.input_jsonl, lines=True)
    output_summaries_beams = []

    input_maintexts = input_ref["maintext"]
    input_ids = input_ref["id"].astype(str)
    model.to("cuda")

    progress_bar = tqdm(range(len(input_maintexts)))

    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        return preds

    for each_text in input_maintexts:
        each_id = tokenizer.encode(("summarize: " + each_text), return_tensors="pt", max_length=256, truncation=True).to("cuda")

        try:
            output_summaries_beams.append( tokenizer.decode(model.generate(each_id, max_length=32, num_beams=20, no_repeat_ngram_size=5)[0], skip_special_tokens=True) )
            progress_bar.update(1)
        except Exception as e:
            print(f"Exception occurred for text: {each_text}")
            print(f"Exception details: {str(e)}")

    output_summaries_beams = postprocess_text(output_summaries_beams)
    df = pd.DataFrame({"title":output_summaries_beams, "id":input_ids})
    df.to_json(args.output_jsonl, orient="records",lines=True)

if __name__ == "__main__":
    main()

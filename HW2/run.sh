summarization_model_folder="summarize_pdtbs_2_pdebs_2_lr_5e5_gas_16_beam_30_epoch_10_cosine"
data_folder="data"

echo "****************Start Unzipping files****************"
unzip "$summarization_model_folder.zip" -d .
unzip "$data_folder.zip" -d .
echo "****************Finish Unzipping files****************"

python inference.py --input_jsonl ${1} --output_jsonl ${2}
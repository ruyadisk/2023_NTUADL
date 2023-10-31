paragraph_model_folder="Paragraph_Selection_pdtbs_4_gas_32_epochs_3"
span_model_folder="Span_Selection_pdtbs_2_gas_32_epochs_3"
data_folder="datas"

echo "****************Start Unzipping files****************"
unzip "$paragraph_model_folder.zip" -d .
unzip "$span_model_folder.zip" -d .
unzip "$data_folder.zip" -d .
echo "****************Finish Unzipping files****************"

python inference.py --context_file ${1} --test_json_file ${2} --output_csv_file ${3} --per_device_eval_batch_size 32
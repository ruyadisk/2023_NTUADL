adapter_folder="adapter_checkpoint"

echo "****************Start Unzipping files****************"
unzip "$adapter_folder.zip" -d .
echo "****************Finish Unzipping files****************"

python inference.py --base_model_path ${1} --peft_path ${2} --test_data_path ${3} --output_path ${4}
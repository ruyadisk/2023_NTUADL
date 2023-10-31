## Step by step instruction
### Enviroment
#### My GPU:
##### 4 * 2080ti (12GB ram per GPU)
#### My CPU:
##### Intel(R) Xeon(R) Silver 4110 CPU @ 2.10GHz

### Paragraph Selection
#### Utils
##### convert2swag.py, run_swag_no_trainer.py
#### Command
##### Convert data to swag dataset format.
##### `python convert2swag.py` 
* ##### Notice that the location you finish conversion. 
##### Train
##### `TOKENIZER_PARALLELISM=false accelerate launch run_swag_no_trainer.py --train_file <location_of_training_set> --validation_file <location_of_validation_set> --max_seq_length 512 --model_name_or_path bert-base-chinese --per_device_train_batch_size 4 --learning_rate 3e-5 --gradient_accumulation_steps 32 --num_train_epochs 3 --output_dir ./Paragraph_Selection_pdtbs_4_gas_32_epochs_3 --with_tracking --report_to wandb --push_to_hub` 

### Span Selection
#### Utils
##### convert2squad.py, run_qa_no_trainer.py, utils_qa.py
#### Command
##### Convert data to squad dataset format.
##### `python convert2squad.py` 
* ##### Notice that the location you finish conversion.
##### Train
##### `TOKENIZER_PARALLELISM=false accelerate launch run_qa_no_trainer.py --train_file <location_of_training_set> --validation_file <location_of_validation_set> --max_seq_length 512 --model_name_or_path bert-base-chinese --per_device_train_batch_size 1 --learning_rate 3e-5 --gradient_accumulation_steps 32 --num_train_epochs 5 --output_dir ./Scratch_bert_base_Span_Selection_pdtbs_2_gas_32_epochs_5 --with_tracking --report_to wandb --weight_decay 1e-3 --lr_scheduler_type polynomial --push_to_hub`

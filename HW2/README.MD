## Step by step instruction
### Enviroment
#### My GPU:
* ##### NVIDIA GeForce RTX 2080 SUPER 8GB
#### My CPU:
* ##### Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz

### Summarization
#### Utils:
* ##### run_summarization_no_trainer.py, compare_strategy.py
#### Command:
##### Train
##### `python run_summarization_no_trainer.py --train_file <your_training_set> --validation_file <your_validation_set> --max_source_length 256 --max_target_length 64 --val_max_target_length 32 --source_prefix summarization: --pad_to_max_length --model_name_or_path google/mt5-small --text_column <your_text_column> --summary_column <your_summary_column> --per_device_train_batch_size 2 --per_device_eval_batch_size 2 --learning_rate 5e-5 --weight_decay 2e-5 --num_train_epochs 10 --gradient_accumulation_steps 16 --lr_scheduler_type cosine --num_warmup_steps 0 --num_beams 30 --output_dir <yout_output_dir> --seed 42 --with_tracking`
 
* ##### Notice that the location of model you finish training. 

##### Compare each strategy's performance
##### `python compare_strategy.py`

* ##### You should modify the code in compare_strategy.py, checking the locations of files, tokenizer and model.
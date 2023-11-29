## Step by step instruction
### Enviroment
#### My GPU:
* ##### NVIDIA GeForce RTX 2080 SUPER 8GB
#### My CPU:
* ##### Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz

### Qlora Finetuning & Evaluation
#### Utils:
* ##### train.sh, train.py, eval.sh, my_ppl_for_all_strategy.py
#### Command:
* ##### Qlora Finetuning
    * ```bash train.sh <path_to_base_model> <path_to_training_dataset> <path_peft_output>```
* ##### Evaluation
    * ```bash eval.sh <path_to_base_model> <path_to_peft> <path_to_test_dataset>```


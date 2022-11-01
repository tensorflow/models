# FFF-NER tf ver.
First, we need to convert a model to our encoder structure, using `./utils/convert_checkpoint`. This will create a 
bert-uncased tf encoder from huggingface's bert-base-uncased model.  
Then, we create the data with `./utils/create_data.py`. A sample dataset can be found in 
`https://github.com/ZihanWangKi/fffner/tree/main/dataset`. If you move the dataset under this folder, you may run
`python3 ./utils/create_data.py dataset/ conll2003`.

Finally, to train the model, you may run
```bash
PYTHONPATH=$PYTHONPATH:/path/to/model/garden \
    python3 train.py \
    --experiment=fffner/ner \
    --config_file=experiments/base_conll2003.yaml \
    --params_override="task.train_data.input_path=conll2003_few_shot_5_0.tf_record,task.validation_data.input_path=conll2003_test.tf_record,runtime.distribution_strategy=tpu,task.init_checkpoint=bert-uncased" \
    --tpu=local \
    --model_dir=tmp_model \
    --mode=train_and_eval
```

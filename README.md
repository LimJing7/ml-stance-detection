# ml-stance-detection

Unless otherwise stated, English datasets are from [mdl-stance-robustness](https://github.com/UKPLab/mdl-stance-robustness) \
NLPCC dataset downloaded from http://tcci.ccf.org.cn/conference/2016/pages/page05_evadata.html \
I am using the Task A Annotated Training Data from stance detection in chinese microblogs task

## Sample Command
CUDA_VISIBLE_DEVICES=3 python run_classify.py --data_dir ~/stance_datasets --model_type xlmr --model_name_or_path microsoft/infoxlm-base --output_dir ../ml-stance-vast-attempt1 --task_name stance --do_train --do_eval --evaluate_during_training --eval_during_train_on_dev --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --learning_rate 5e-6 --adam_epsilon 1e-8 --num_train_epochs 20 --overwrite_output_dir --loss_fn bce --logging_steps 500 --save_steps 500 --save_only_best_checkpoint --train_dataset vast

## Preliminary Results

### Chinese
| Attribute              | Value |
|------------------------|-------|
| Train Dataset          | nlpcc |
| Test Dataset           | nlpcc |
| Learning Rate          |  5e-6 |
| Number of Train Epochs |    20 |
| Logging / Save steps   |   500 |
| Accuracy               |  0.63 |
| Macro F1               |  0.56 |

### English
| Attribute              | Value |
|------------------------|-------|
| Train Dataset          |   arc |
| Test Dataset           |   arc |
| Learning Rate          |  5e-6 |
| Number of Train Epochs |    20 |
| Logging / Save steps   |   500 |
| Accuracy               |  0.62 |
| Macro F1               |  0.31 |

| Attribute              | Value  |
|------------------------|--------|
| Train Dataset          | argmin |
| Test Dataset           | argmin |
| Learning Rate          |   5e-6 |
| Number of Train Epochs |     20 |
| Logging / Save steps   |    500 |
| Accuracy               |   0.71 |
| Macro F1               |   0.51 |

| Attribute              | Value |
|------------------------|-------|
| Train Dataset          |  fnc1 |
| Test Dataset           |  fnc1 |
| Learning Rate          |  5e-6 |
| Number of Train Epochs |    20 |
| Logging / Save steps   |   500 |
| Accuracy               |  0.64 |
| Macro F1               |  0.46 |

| Attribute              | Value |
|------------------------|-------|
| Train Dataset          |  iac1 |
| Test Dataset           |  iac1 |
| Learning Rate          |  1e-6 |
| Number of Train Epochs |    20 |
| Logging / Save steps   |   500 |
| Accuracy               |  0.48 |
| Macro F1               |  0.40 |

| Attribute              | Value |
|------------------------|-------|
| Train Dataset          | ibmcs |
| Test Dataset           | ibmcs |
| Learning Rate          |  5e-6 |
| Number of Train Epochs |    50 |
| Logging / Save steps   |   500 |
| Accuracy               |  0.71 |
| Macro F1               |  0.71 |

| Attribute              | Value       |
|------------------------|-------------|
| Train Dataset          | perspectrum |
| Test Dataset           | perspectrum |
| Learning Rate          |        5e-6 |
| Number of Train Epochs |          20 |
| Logging / Save steps   |         500 |
| Accuracy               |        0.83 |
| Macro F1               |        0.83 |

| Attribute              | Value         |
|------------------------|---------------|
| Train Dataset          | semeval2016t6 |
| Test Dataset           | semeval2016t6 |
| Learning Rate          |          5e-5 |
| Number of Train Epochs |            20 |
| Logging / Save steps   |           500 |
| Accuracy               |          0.73 |
| Macro F1               |          0.23 |

| Attribute              | Value  |
|------------------------|--------|
| Train Dataset          | snopes |
| Test Dataset           | snopes |
| Learning Rate          |   5e-6 |
| Number of Train Epochs |     20 |
| Logging / Save steps   |    500 |
| Accuracy               |   0.80 |
| Macro F1               |   0.75 |

| Attribute              | Value       |
|------------------------|-------------|
| Train Dataset          | twitter2015 |
| Test Dataset           | twitter2015 |
| Learning Rate          |        5e-6 |
| Number of Train Epochs |          20 |
| Logging / Save steps   |         500 |
| Accuracy               |        0.69 |
| Macro F1               |        0.65 |

| Attribute              | Value       |
|------------------------|-------------|
| Train Dataset          | twitter2017 |
| Test Dataset           | twitter2017 |
| Learning Rate          |        5e-6 |
| Number of Train Epochs |          20 |
| Logging / Save steps   |         500 |
| Accuracy               |        0.69 |
| Macro F1               |        0.67 |

| Attribute              | Value |
|------------------------|-------|
| Train Dataset          |  vast |
| Test Dataset           |  vast |
| Learning Rate          |  5e-6 |
| Number of Train Epochs |    20 |
| Logging / Save steps   |   500 |
| Accuracy               |  0.64 |
| Macro F1               |  0.49 |

### Crosslingual
| Attribute              | Value |
|------------------------|-------|
| Train Dataset          |   arc |
| Test Dataset           | nlpcc |
| Learning Rate          |  5e-6 |
| Number of Train Epochs |    20 |
| Logging / Save steps   |   500 |
| Accuracy               |  0.40 |
| Macro F1               |  0.21 |

| Attribute              | Value                                                              |
|------------------------|--------------------------------------------------------------------|
| Train Dataset          | arc, argmin, fnc1, iac1, ibmcs, perspectrum, semeval2016t6, snopes |
| Test Dataset           |                                                              nlpcc |
| Learning Rate          |                                                               3e-5 |
| Number of Train Epochs |                                                                  5 |
| Logging / Save steps   |                                                                500 |
| Accuracy               |                                                               0.41 |
| Macro F1               |                                                               0.16 |

| Attribute              | Value                                                              |
|------------------------|--------------------------------------------------------------------|
| Train Dataset          | arc, argmin, fnc1, iac1, ibmcs, perspectrum, semeval2016t6, snopes |
| Test Dataset           |                                                              nlpcc |
| Learning Rate          |                                                               1e-5 |
| Number of Train Epochs |                                                                  5 |
| Logging / Save steps   |                                                                500 |
| Accuracy               |                                                               0.50 |
| Macro F1               |                                                               0.28 |

| Attribute              | Value                                                              |
|------------------------|--------------------------------------------------------------------|
| Train Dataset          | arc, argmin, fnc1, iac1, ibmcs, perspectrum, semeval2016t6, snopes |
| Test Dataset           |                                                              nlpcc |
| Learning Rate          |                                                               5e-6 |
| Number of Train Epochs |                                                                  5 |
| Logging / Save steps   |                                                                500 |
| Accuracy               |                                                               0.47 |
| Macro F1               |                                                               0.23 |


## Datasets
### NLPCC
NLPCC splits are generated by me and topics are shared between train, dev and test

|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |   451   |      0     |    449   |      218     |
| Dev   |   161   |      0     |    140   |      73      |
| Test  |   597   |      0     |    575   |      321     |

### ARC
ARC splits are found from the mdl repo and topics are shared

|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |   1220  |     786    |   1094   |     9282     |
| Dev   |   182   |     118    |    163   |     1388     |
| Test  |   372   |     179    |    334   |     2674     |

### ARGMIN
ArgMin are found from the mdl repo and 5 topics are used for training, 1 topic for dev and 2 topics for test

|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |  3690   |     0      |   3155   |      0       |
| Dev   |  1111   |     0      |   457    |      0       |
| Test  |  1394   |     0      |   1332   |      0       |

### IAC1
IAC1 splits are found from the mdl repo w/o overlap in topics between the 3 sets
|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |  1432   |    371     |   2424   |      0       |
| Dev   |   117   |    113     |   224    |      0       |
| Test  |   395   |     96     |   433    |      0       |

### IBMCS
IBMCS splits are found from the mdl repo

|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |   373   |     0      |   562    |      0       |
| Dev   |   41    |     0      |    63    |      0       |
| Test  |   655   |     0      |   700    |      0       |

### FNC1
FNC1 splits are found from the mdl repo

|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |   714   |    7573    |   3126   |    31063     |
| Dev   |   126   |    1336    |   552    |     5482     |
| Test  |   697   |    4464    |   1903   |    18349     |


### Perspectrum
Perspectrum splits are found from the mdl repo and topics are shared

|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |   3379  |      0     |   3599   |       0      |
| Dev   |   1024  |      0     |   1047   |       0      |
| Test  |   1302  |      0     |   1471   |       0      |

### Semeval 2016 task 6
Semeval 2016 task 6 splits are found from the mdl repo and topics are shared

|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |   1191  |     657    |    649   |       0      |
| Dev   |   204   |     109    |    104   |       0      |
| Test  |   715   |     230    |    304   |       0      |

### Snopes
Snopes splits are found from the mdl repo

|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |  3899   |     0      |  10517   |      0       |
| Dev   |   515   |     0      |   1353   |      0       |
| Test  |   678   |     0      |   2476   |      0       |

### Twitter 2015
Twitter 2015 data are found from [TomBERT](https://github.com/jefferyYu/TomBERT)

|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |   368   |    1883    |   928    |      0       |
| Dev   |   149   |    670     |   303    |      0       |
| Test  |   113   |    607     |   317    |      0       |


### Twitter 2017
Twitter 2017 data are found from [TomBERT](https://github.com/jefferyYu/TomBERT)

|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |   416   |    1638    |   1508   |      0       |
| Dev   |   144   |    517     |   515    |      0       |
| Test  |   168   |    573     |   493    |      0       |


### VAST
Vast splits are found from [zero-shot-stance](https://github.com/emilyallaway/zero-shot-stance)

|       | Against | Discussing | In Favor | Unrelated To |
|-------|:-------:|:----------:|:--------:|:------------:|
| Train |  5595   |    2555    |   5327   |      0       |
| Dev   |   684   |    697     |   681    |      0       |
| Test  |  1018   |    1044    |   944    |      0       |



## Future Work
- Add more datasets
- Run some cross-lingual evaluations

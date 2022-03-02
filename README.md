# ml-stance-detection

Unless otherwise stated, English datasets are from [mdl-stance-robustness](https://github.com/UKPLab/mdl-stance-robustness) \
NLPCC dataset downloaded from http://tcci.ccf.org.cn/conference/2016/pages/page05_evadata.html \
I am using the Task A Annotated Training Data from stance detection in chinese microblogs task
Arabic News Stance (ans) corpus downloaded from https://github.com/latynt/ans

## Sample Command
CUDA_VISIBLE_DEVICES=3 python run_classify.py --data_dir ~/stance_datasets --model_type xlmr --model_name_or_path microsoft/infoxlm-base --output_dir ../ml-stance-vast-attempt1 --task_name stance --do_train --do_eval --evaluate_during_training --eval_during_train_on_dev --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --learning_rate 5e-6 --adam_epsilon 1e-8 --num_train_epochs 20 --overwrite_output_dir --loss_fn bce --logging_steps 500 --save_steps 500 --save_only_best_checkpoint --train_dataset vast

## Preliminary Results

### Arabic
| Attribute              | Value |
|------------------------|-------|
| Train Dataset          |   ans |
| Test Dataset           |   ans |
| Mlm                    | false |
| Learning Rate          |  5e-6 |
| Number of Train Epochs |    20 |
| Logging / Save steps   |   500 |
| Accuracy               |  0.80 |
| Macro F1               |  0.79 |

| Attribute              | Value |
|------------------------|-------|
| Train Dataset          |   ans |
| Test Dataset           |   ans |
| Mlm                    |  true |
| Learning Rate          |  5e-6 |
| Number of Train Epochs |    20 |
| Logging / Save steps   |   500 |
| Accuracy               |  0.57 |
| Macro F1               |  0.43 |


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
| Mlm                    | false |
| Learning Rate          |  5e-6 |
| Number of Train Epochs |    20 |
| Logging / Save steps   |   500 |
| Accuracy               |  0.62 |
| Macro F1               |  0.31 |

| Attribute              | Value |
|------------------------|-------|
| Train Dataset          |   arc |
| Test Dataset           |   arc |
| Mlm                    |  true |
| Learning Rate          |  5e-6 |
| Number of Train Epochs |    20 |
| Logging / Save steps   |   500 |
| Accuracy               |  0.82 |
| Macro F1               |  0.62 |

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
| Mlm                    | false |
| Learning Rate          |  5e-6 |
| Number of Train Epochs |    20 |
| Logging / Save steps   |   500 |
| Accuracy               |  0.99 |
| Macro F1               |  0.97 |

| Attribute              | Value |
|------------------------|-------|
| Train Dataset          |  fnc1 |
| Test Dataset           |  fnc1 |
| Mlm                    |  true |
| Learning Rate          |  5e-6 |
| Number of Train Epochs |    20 |
| Logging / Save steps   |   500 |
| Accuracy               |  0.84 |
| Macro F1               |  0.74 |

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

| Attribute              | Value                                                                                                  |
|------------------------|--------------------------------------------------------------------------------------------------------|
| Train Dataset          | arc, argmin, fnc1, iac1, ibmcs, perspectrum, semeval2016t6,<br> snopes, twitter2015, twitter2017, vast |
| Test Dataset           |                                                                                                  nlpcc |
| Learning Rate          |                                                                                                   1e-5 |
| Number of Train Epochs |                                                                                                      5 |
| Logging / Save steps   |                                                                                                    500 |
| Accuracy               |                                                                                                   0.49 |
| Macro F1               |                                                                                                   0.34 |

| Attribute              | Value                                                                                                  |
|------------------------|--------------------------------------------------------------------------------------------------------|
| Train Dataset          | arc, argmin, fnc1, iac1, ibmcs, perspectrum, semeval2016t6,<br> snopes, twitter2015, twitter2017, vast |
| Test Dataset           |                                                                                                  nlpcc |
| Learning Rate          |                                                                                                   1e-5 |
| Number of Train Epochs |                                                                                                      5 |
| Logging / Save steps   |                                                                                                    500 |
| Negative Samples (wo wiki syns) |                                                                                             2 |
| Accuracy               |                                                                                                   0.46 |
| Macro F1               |                                                                                                   0.41 |

| Attribute              | Value                                                                                                  |
|------------------------|--------------------------------------------------------------------------------------------------------|
| Train Dataset          | arc, argmin, fnc1, iac1, ibmcs, perspectrum, semeval2016t6,<br> snopes, twitter2015, twitter2017, vast |
| Test Dataset           |                                                                                       translated nlpcc |
| Learning Rate          |                                                                                                   1e-5 |
| Number of Train Epochs |                                                                                                      5 |
| Logging / Save steps   |                                                                                                    500 |
| Negative Samples (wo wiki syns) |                                                                                             2 |
| Accuracy               |                                                                                                   0.41 |
| Macro F1               |                                                                                                   0.36 |

| Attribute              | Value                                                                                                  |
|------------------------|--------------------------------------------------------------------------------------------------------|
| Train Dataset          | arc, argmin, fnc1, iac1, ibmcs, perspectrum, semeval2016t6,<br> snopes, twitter2015, twitter2017, vast |
| Test Dataset           |                                                                                                  nlpcc |
| Learning Rate          |                                                                                                   1e-5 |
| Number of Train Epochs |                                                                                                      5 |
| Logging / Save steps   |                                                                                                    500 |
| Negative Samples       |                                                                                                      2 |
| Accuracy               |                                                                                                   0.47 |
| Macro F1               |                                                                                                   0.41 |


| Attribute              | Value                                                                                                  |
|------------------------|--------------------------------------------------------------------------------------------------------|
| Train Dataset          | arc, argmin, fnc1, iac1, ibmcs, perspectrum, semeval2016t6,<br> snopes, twitter2015, twitter2017, vast |
| Test Dataset           |                                                                                       translated nlpcc |
| Learning Rate          |                                                                                                   1e-5 |
| Number of Train Epochs |                                                                                                      5 |
| Logging / Save steps   |                                                                                                    500 |
| Negative Samples       |                                                                                                      2 |
| Accuracy               |                                                                                                   0.51 |
| Macro F1               |                                                                                                   0.40 |

## Future Work
- Add more datasets
- Run some cross-lingual evaluations

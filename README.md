# Multilingual Stance Detection

Unless otherwise stated, English datasets are from [mdl-stance-robustness](https://github.com/UKPLab/mdl-stance-robustness). \
NLPCC dataset downloaded from http://tcci.ccf.org.cn/conference/2016/pages/page05_evadata.html. \
I am using the "Task A Annotated Training Data" from the "Stance Detection in Chinese Microblogs" task. \
Arabic News Stance (ans) corpus downloaded from https://github.com/latynt/ans.

## Sample Command
CUDA_VISIBLE_DEVICES=3 python run_classify.py --data_dir ~/stance_datasets --model_type xlmr --model_name_or_path microsoft/infoxlm-base --output_dir ../ml-stance-vast-attempt1 --task_name stance --do_train --do_eval --evaluate_during_training --eval_during_train_on_dev --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --learning_rate 5e-6 --adam_epsilon 1e-8 --num_train_epochs 20 --overwrite_output_dir --loss_fn bce --logging_steps 500 --save_steps 500 --save_only_best_checkpoint --train_dataset vast

## Approach
We use the following pattern: 'The stance of the following \<text\>is \<mask_token\> the \<topic\>'.
An example after we put it into the pattern looks like this following:
```
The stance of the following And what will police do with this information? Try to make sure the patient has no weapons? (No access to local gun shows or other private sellers?) The patient has committed no crime so he (yes, he) cannot be arrested. In exchange for this, we have ensured that the patient will end therapy and never seek help again. Is this a good trade for society? I think not. is \<mask_token\> the No, the laws should
not be toughened.
```
The model is given the above as input and asked to predict the mask token. We take the last hidden state at the mask location and dot product it with the mean embeddings for each of the labels (can be multiple tokens) to find the closest label.

If the tokenized text and topic are too long, we truncate them to be as even as possible.
That is we truncate the longer input first. When they are both at the same length, we truncate one token at a time from each.

We use a Binary Cross Entropy (BCE) loss for each label as we do not want to downweigh synonyms. This is in contrast to cross-entropy which down-weighs all labels other than the right one.

## Options
### Negative Sampling
Uses the synonyms for negative labels as negative samples.
Synonyms are drawn from wordnet and wiktionary.

### Masked Language Modelling (MLM)
Includes MLM loss into the objective.
Uses an alpha parameter to weigh the loss.

### Same vs Different Labels
The datasets are either using their own labels or they can all share one set of labels. \
To use the same labels switch to branch same_labels.

## Dataset weighting
Equal: No weighting \
Multinomial: weigh by the following formula \
<img src="https://render.githubusercontent.com/render/math?math=q_i = \frac{p_i^\alpha}{\sum_{j=1}^N p_j^\alpha}"> where <img src="https://render.githubusercontent.com/render/math?math=p_i = \frac{n_i}{\sum_{k=1}^N n_k}"> and <img src="https://render.githubusercontent.com/render/math?math=n_i"> is the number of examples in that dataset


## Results
For the results on this page, we are only listing results that come from training on all english datasets and evaluating on the nlpcc development set. We also evaluated on the nlpcc development set after translating them from chinese to english using google translate as a baseline (trans_nlpcc).

For single dataset results, refer to [single_dataset_results.md](./single_dataset_results.md). However, note that some of them might be outdated and trained using previous versions of the code.

### Hyperparameters
We have included the hyperparameters we used for the below experiments here:
<table>
<tr><td> Learning Rate </td><td> 1e-5 </td></tr>
<tr><td> Adam Epsilon </td><td> 1e-8 </td></tr>
<tr><td> Number of Train Epochs </td><td> 5 </td></tr>
<tr><td> Logging / Save Steps </td><td> 500 </td></tr>
<tr><td> MLM alpha </td><td> 0.5 </td></tr>
<tr><td> Loss function </td><td> BCE Loss </td></tr>
<tr><td> Dataset Weighting alpha </td><td> 0.3 </td></tr>
</table>


### Different Labels
|             | base | with mlm | with 2 negative | with mlm and 2 negative |
|-------------|:----:|:--------:|:---------------:|:-----------------------:|
| nlpcc       | 0.43 |   0.42   |       0.43      |           0.42          |
| trans_nlpcc | 0.39 |   0.40   |       0.39      |           0.42          |


### Same Labels
|             | base | with mlm | with 2 negative | with mlm and 2 negative |
|-------------|:----:|:--------:|:---------------:|:-----------------------:|
| nlpcc       | 0.42 |   0.42   |       0.42      |           0.42          |
| trans_nlpcc | 0.38 |   0.31   |       0.38      |           0.35          |

### Using Multinomial
|             | base | with mlm | with 2 negative | with mlm and 2 negative |
|-------------|:----:|:--------:|:---------------:|:-----------------------:|
| nlpcc       | 0.47 |          |                 |                         |
| trans_nlpcc | 0.41 |          |                 |                         |



## Future Work
- Add more datasets
- Use different weights for different datasets
- End-task aware training
- Meta end-task aware training


## Dataset Statistics
Statistics for each dataset has been moved to a separate [file](./datasets_statistics.md).

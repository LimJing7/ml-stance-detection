# Multilingual Stance Detection

We are tackling the problem of stance detection. Given a text and a topic, we want to predict the stance of the text towards the topic.
In this repo, we are interested in the problem of zero-shot language transfer for stance detection. That is we train on stance detection using English datasets before performing evaluation on non-English datasets.

## Datasets Used
Unless otherwise stated, English datasets are from [mdl-stance-robustness](https://github.com/UKPLab/mdl-stance-robustness). \
NLPCC dataset downloaded from http://tcci.ccf.org.cn/conference/2016/pages/page05_evadata.html. \
I am using the "Task A Annotated Training Data" from the "Stance Detection in Chinese Microblogs" task. \
Arabic News Stance (ans) corpus downloaded from https://github.com/latynt/ans.

Statistics for each dataset can be found in a separate [file](./datasets_statistics.md).

## Details

### Approach

We are following the method found in [Hardalov et al.] (https://arxiv.org/abs/2109.06050).

We use the following pattern: 'The stance of the following \<text\>is \<mask_token\> the \<topic\>'.
An example after we put it into the pattern looks like this following:

> The stance of the following And what will police do with this information? Try to make sure the patient has no weapons? (No access to local gun shows or other private sellers?) The patient has committed no crime so he (yes, he) cannot be arrested. In exchange for this, we have ensured that the patient will end therapy and never seek help again. Is this a good trade for society? I think not. is \<mask_token\> the No, the laws should not be toughened.

The model is given the above as input and asked to predict the mask token. We take the last hidden state at the mask location and dot product it with the mean embeddings for each of the labels (can be multiple tokens) to find the closest label.

If the tokenized text and topic are too long, we truncate them to be as even as possible.
That is we truncate the longer input first. When they are both at the same length, we truncate one token at a time from each.

We use a Binary Cross Entropy (BCE) loss for each label as we do not want to downweigh synonyms. Using the BCE, we can make it upweigh the correct labels and only downweigh the wrong labels. This is in contrast to cross-entropy which down-weighs all labels other than the right one.

### Options
#### Negative Sampling
Uses the synonyms for negative labels as negative samples.
Synonyms are drawn from wordnet and wiktionary.

#### Masked Language Modelling (MLM)
Includes MLM loss into the objective.
Uses an alpha parameter to weigh the loss.

#### Same vs Different Labels
The datasets are either using their own labels. For example:
- arc:
  - disagreeing with
  - discussing
  - agreeing with
  - unrelated to
- twitter2017:
  - negative
  - positive
  - neutral

Or they can all share one set of labels:
- against
- discussing
- in favour
- unrelated to

To use the same labels switch to branch same_labels.

#### Dataset weighting
We can add a weighting term to the loss function for each dataset to change the influence of each dataset.
- Equal
  -  Equal weighting among all datasets results in a lower influence for smaller datasets.
- Scaled
  - Compute a scaled weighting from the size of the datasets.
  - Using the following scaling equation, we are able to increase the influence of the larger datasets.
  - Let <img src="https://render.githubusercontent.com/render/math?math=n_i"> be the number of examples in that dataset and <img src="https://render.githubusercontent.com/render/math?math=0\le\alpha\le1"> be a parameter we choose.
  1. Compute <img src="https://render.githubusercontent.com/render/math?math=p_i = \frac{n_i}{\sum_{k=1}^N n_k}">
  2. Compute the dataset weights <img src="https://render.githubusercontent.com/render/math?math=q_i = \frac{p_i^\alpha}{\sum_{j=1}^N p_j^\alpha}">
  3. Multiply the weights by the number of datasets used for training
- Inverse Scaled
  - Compute the dataset weights in the same way as in "scaled" but instead of using that as a multiplier, use it as a divisor.
  - This allows us to upweigh the influence of the smaller datasets while still letting the larger datasets have a greater influence.
- Uncorrected Scaled
  - Compute dataset weights in the same way as in "scaled" but without step 3.

#### Robust Training
We adapted the methods from [Improving Zero-Shot Cross-Lingual Transfer Learning via Robust Training](https://aclanthology.org/2021.emnlp-main.126.pdf)\(Huang et. al.\) to improve the robustness of the trained embeddings.

We tried two variations of the Randomized Smoothing method, Random Perturbation (RS-RP) and Data Augmentation (RS-DA).
In RS-RP, we sample a noise vector and add it to the embeddings before performing classification. Whereas in RS-DA, we used a synonym set to replace the words in the train set with their synonyms and use these as the augmented training examples. We modified the code from their [github](https://github.com/uclanlp/Robust-XLT) and also took the synonyms set from there.

#### TARTAN
We adapted the meta-tartan method from [SHOULD WE BE Pre-TRAINING? EXPLORING END-TASK AWARE TRAINING IN LIEU OF CONTINUED PRE-TRAINING](https://openreview.net/pdf?id=2bO2x8NAIMB) \(Dery et. al.\). Instead of using the meta-training to learn weights to modulate between task losses, we use it to modulate between the datasets that we are using. Due to the amount of VRAM required, we were unable to train the MT-TARTAN model on all 11 English datasets. Hence, we decided to just train on 5 datasets. For the first experiment, we picked the 5 datasets that we felt will be the most useful. For the next 2 experiments, we ran some simple one dataset experiments to pick the top two most useful datasets. We then ran further experiments by adding one dataset at a time to the training. This allowed us to put 3 more datasets and train them with the two previously selected ones.

We then contrasted the same models trained without TARTAN to see what advantage does TARTAN bring for us.

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

## Training Setup
### Datasets
Put all datasets into a data_dir and name the files as \<dataset\>-\<split\>.json

### Sample Command
CUDA_VISIBLE_DEVICES=3 python run_classify.py --data_dir ~/stance_datasets --model_type xlmr --model_name_or_path microsoft/infoxlm-base --output_dir ../ml-stance-vast-attempt1 --task_name stance --do_train --do_eval --evaluate_during_training --eval_during_train_on_dev --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 16 --learning_rate 5e-6 --adam_epsilon 1e-8 --num_train_epochs 20 --overwrite_output_dir --loss_fn bce --logging_steps 500 --save_steps 500 --save_only_best_checkpoint --train_dataset vast

## Results
For the results on this page, we are only listing results that come from training on all english datasets and evaluating on the nlpcc development set. We also evaluated on the nlpcc development set after translating them from chinese to english using google translate as a baseline (trans_nlpcc).

For results from supervised training on in-domain data, refer to [single_dataset_results.md](./single_dataset_results.md). However, note that some of them might be outdated and trained using previous versions of the code.

For comparison, the macro-F1 in Hardalov et al. is 0.458.


### Different Labels
|             | base | with mlm | with 2 negative | with mlm and 2 negative |
|-------------|:----:|:--------:|:---------------:|:-----------------------:|
| nlpcc       | 0.43 |   0.42   |       0.49      |           0.42          |
| trans_nlpcc | 0.39 |   0.40   |       0.42      |           0.42          |

### Same Labels
|             | base | with mlm | with 2 negative | with mlm and 2 negative |
|-------------|:----:|:--------:|:---------------:|:-----------------------:|
| nlpcc       | 0.42 |   0.42   |       0.42      |           0.42          |
| trans_nlpcc | 0.38 |   0.31   |       0.38      |           0.35          |


### Using the uncorrected scaled dataset weighing scheme
|             | base | with mlm | with 2 negative | with mlm and 2 negative |
|-------------|:----:|:--------:|:---------------:|:-----------------------:|
| nlpcc       | 0.47 |   0.43   |       0.47      |           0.45          |
| trans_nlpcc | 0.41 |   0.39   |       0.43      |           0.39          |


### Compare different dataset weighing schemes
Using 2 negative
|             |   Equal  | Scaled | Inverse Scaled | Uncorrected Scaled | Scaled with smaller LR | Random |
|-------------|:--------:|:------:|:--------------:|:------------------:|:----------------------:|:------:|
| nlpcc       |   0.49   |  0.47  |      0.47      |        0.47        |          0.35          |  0.46  |
| trans_nlpcc |   0.40   |  0.39  |      0.44      |        0.43        |          0.32          |  0.43  |


### Compare different robust training schemes
|             | RS-RP (0.01) | RS-RP (0.1) | RS-DA |   |
|-------------|:------------:|:-----------:|:-----:|:-:|
| nlpcc       |   **0.50**   |     0.48    |  0.48 |   |
| trans_nlpcc |     0.43     |     0.41    |  0.43 |   |

### Different Masking Rate for MLM
|             | 0.15 |  0.3 |  0.4 |
|-------------|:----:|:----:|:----:|
| nlpcc       | 0.43 | 0.45 | 0.46 |
| trans_nlpcc | 0.41 | 0.36 | 0.37 |

### Tartan
|                | arc argmin fnc1 twitter2015 twitter2017 <br> 20 epochs | argmin semeval2016t6 twitter2015 twitter2017 vast <br> 5 epochs | argmin semeval2016t6 twitter2015 twitter2017 vast <br> 20 epochs | argmin semeval2016t6 twitter2015 twitter2017 vast <br> 5 epochs + rs_rp | argmin semeval2016t6 twitter2015 twitter2017 vast <br> 20 epochs + rs_rp |
|----------------|:------------------------------------------------------:|:---------------------------------------------------------------:|:----------------------------------------------------------------:|:-----------------------------------------------------------------------:|:------------------------------------------------------------------------:|
| with tartan    |                          0.48                          |                               0.46                              |                               0.48                               |                                   0.46                                  |                                   0.47                                   |
| without tartan |                          0.47                          |                               0.43                              |                               0.46                               |                                   0.46                                  |                                   0.49                                   |
- Add more datasets
- End-task aware training
- Meta end-task aware training

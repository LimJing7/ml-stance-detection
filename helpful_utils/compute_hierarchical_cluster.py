
import argparse
import bisect
import glob
import logging
import os
import pickle
import random
import statistics

import numpy as np
import sklearn
from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch
from tqdm import tqdm, trange

from transformers import (
  WEIGHTS_NAME,
  AdamW,
  BertConfig,
  BertForMaskedLM,
  BertTokenizer,
  XLMConfig,
  XLMWithLMHeadModel,
  XLMTokenizer,
  XLMRobertaConfig,
  XLMRobertaTokenizer,
  XLMRobertaModel,
  get_linear_schedule_with_warmup,
)

from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from processors.nusax import NusaXProcessor


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


languages = ['acehnese',
             'balinese',
             'banjarese',
             'buginese',
             'english',
             'indonesian',
             'javanese',
             'madurese',
             'minangkabau',
             'ngaju',
             'sudanese',
             'toba_batak']

tokenizer = XLMRobertaTokenizer.from_pretrained('microsoft/infoxlm-base')
model = XLMRobertaModel.from_pretrained('microsoft/infoxlm-base').cuda()

avg_lang_emb = {}

for language in languages:
    processor = NusaXProcessor(language)
    examples = processor.get_train_examples('/projects/tir5/users/jinglim/stance_datasets')

    examples_text = [i.text for i in examples]
    tokenized_tensors = tokenizer(examples_text, padding=True, return_tensors='pt')
    dataset = torch.utils.data.TensorDataset(tokenized_tensors['input_ids'], tokenized_tensors['attention_mask'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    for batch in dataloader:
        tokenized_tensors = [i.cuda() for i in batch]
        output = model(input_ids=tokenized_tensors[0], attention_mask=tokenized_tensors[1])['pooler_output']  ## get emb of <s>
        comb_emb = output.sum(dim=0).detach().cpu().numpy()
        try:
            avg_lang_emb[language] += comb_emb
        except KeyError:
            avg_lang_emb[language] = comb_emb

    avg_lang_emb[language] /= len(dataset)

avg_lang_emb_arr = np.array([avg_lang_emb[lang] for lang in languages])


fig, axes = plt.subplots(2,2,tight_layout=True,figsize=(12.8,9.6))
fig.suptitle("Hierarchical Clustering Dendrogram")

for i, linkage in enumerate(['ward', 'complete', 'average', 'single']):
    clustering_model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=linkage)
    clustering_model.fit(avg_lang_emb_arr)
    ax = axes[i//2, i%2]
    plot_dendrogram(clustering_model, labels=languages, leaf_rotation=70, ax=ax)
    ax.set_title(f"Linkage: {linkage}")

# plot the top three levels of the dendrogram
plt.savefig('hierarch.png', bbox_inches='tight')
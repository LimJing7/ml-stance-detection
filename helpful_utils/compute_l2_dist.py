
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
    print(language)
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

l2_dist = {}
for lang in languages:
    en_emb = avg_lang_emb['english']
    lang_emb = avg_lang_emb[lang]
    l2_dist[lang] = np.linalg.norm(en_emb - lang_emb)

pickle.dump(l2_dist, open('nusax_l2.pkl', 'wb'))

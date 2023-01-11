import argparse
from collections import defaultdict
import os
import pickle
import torch
import tqdm
import transformers

from processors.combnlpcc import CombNLPCCProcessor

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='microsoft/infoxlm-base')
parser.add_argument('--data_dir', default='/projects/tir5/users/jinglim/stance_datasets')
parser.add_argument('--mode', choices=['topic', 'label', 'topic_label', 'label+topic'])

args = parser.parse_args()

tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(args.model)
model = transformers.XLMRobertaModel.from_pretrained(args.model)

processor = CombNLPCCProcessor()
examples = processor.get_dev_examples(args.data_dir)

per_group = defaultdict(lambda: torch.zeros(model.config.hidden_size))
per_group_count = defaultdict(int)

for example in tqdm.tqdm(examples):
    text = example.text
    tokenized = torch.tensor(tokenizer([text])['input_ids'])
    emb = model(tokenized)['pooler_output'].detach()
    label = example.label
    topic = example.topic
    if args.mode == 'topic':
        per_group[topic] += emb[0]
        per_group_count[topic] += 1
    elif args.mode == 'label':
        per_group[label] += emb[0]
        per_group_count[label] += 1
    elif args.mode == 'topic_label':
        topic_label = f'{topic}_{label}'
        per_group[topic_label] += emb[0]
        per_group_count[topic_label] += 1
    elif args.mode == 'label+topic':
        per_group[label] += emb[0]
        per_group_count[label] += 1
        per_group[topic] += emb[0]
        per_group_count[topic] += 1

for group in per_group:
    per_group[label] /= per_group_count[label]

per_group = dict(per_group)

with open(f'per_{args.mode}_emb.pkl', 'wb') as f:
    pickle.dump(per_group, f)
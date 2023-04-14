import argparse
from collections import defaultdict
import os
import pickle
import torch
import tqdm
import transformers

from run_classify import PROCESSORS

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='microsoft/infoxlm-base')
parser.add_argument('--data_dir', default='/projects/tir5/users/jinglim/stance_datasets')
parser.add_argument('--dataset', default='comb_nlpcc')
parser.add_argument('--split', choices=['train', 'dev', 'test'], default='dev')
parser.add_argument('--mode', choices=['topic', 'label', 'topic_label', 'label+topic'], required=True)

args = parser.parse_args()

def get_processor(dataset, taskname, seed):
  if dataset.startswith('comb_nlpcc_'):
    count = dataset.split('_')[-1]
    processor = PROCESSORS[taskname]['comb_nlpcc_fs'](count, seed)
  elif dataset.startswith('maj_twitter_bulk_emb-'):
    neighbours = int(dataset.split('-')[1])
    variant = dataset.split('-')[2]
    processor = PROCESSORS[taskname]['maj_twitter_bulk_emb'](neighbours, variant)
  elif dataset.startswith('maj_twitter_v2_bulk_emb-'):
    neighbours = int(dataset.split('-')[1])
    variant = dataset.split('-')[2]
    processor = PROCESSORS[taskname]['maj_twitter_v2_bulk_emb'](neighbours, variant, 2)
  elif dataset.startswith('maj_twitter_bulk_'):
    count = int(dataset.split('_')[-1])
    processor = PROCESSORS[taskname]['maj_twitter_bulk'](count, seed)
  elif dataset.startswith('maj_twitter_tfidf_'):
    count = int(dataset.split('_')[-1])
    processor = PROCESSORS[taskname]['maj_twitter_tfidf'](count, seed)
  elif dataset.startswith('maj_twitter_wiki_cont_filt_rand'):
    count = int(dataset.split('_')[-1])
    processor = PROCESSORS[taskname]['maj_twitter_wiki_cont_filt_rand'](count, seed)
  elif dataset.startswith('nusax_'):
    lang = '_'.join(dataset.split('_')[1:])
    processor = PROCESSORS[taskname]['nusax'](lang)
  elif dataset.startswith('twitter_wiki_cont_3_part'):
    part = int(dataset.split('_')[-1])
    processor = PROCESSORS[taskname]['twitter_wiki_cont_3'](part)
  elif dataset.startswith('xstance_'):
    lang = dataset.split('_')[-1]
    processor = PROCESSORS[taskname]['xstance'](lang)
  elif dataset.startswith('zh_'):
    act_ds = dataset.split('_')[-1]
    processor = PROCESSORS[taskname]['zh'](act_ds)
  else:
    processor = PROCESSORS[taskname][dataset]()
  return processor

if torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cpu')

torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(args.model)
model = transformers.XLMRobertaModel.from_pretrained(args.model).to(device)

processor = get_processor(args.dataset, 'stance', 0)
try:
    examples = processor.get_examples_generator(args.data_dir, split=args.split)
except AttributeError:
    examples = processor.get_examples(args.data_dir, split=args.split)

per_group = defaultdict(lambda: torch.zeros(model.config.hidden_size))
per_group_count = defaultdict(int)

example_batch = []

def process_batch(args, device, tokenizer, model, per_group, per_group_count, example_batch):
    texts = [example.text for example in example_batch]
    tokenizeds = torch.tensor(tokenizer(texts, padding=True, truncation=True, max_length=128)['input_ids']).to(device)
    embs = model(tokenizeds)['pooler_output'].detach().cpu()
    labels = [example.label for example in example_batch]
    topics = [example.topic for example in example_batch]
    for emb, label, topic in zip(embs, labels, topics):
        if args.mode == 'topic':
            per_group[topic] += emb
            per_group_count[topic] += 1
        elif args.mode == 'label':
            per_group[label] += emb
            per_group_count[label] += 1
        elif args.mode == 'topic_label':
            topic_label = f'{topic}_{label}'
            per_group[topic_label] += emb
            per_group_count[topic_label] += 1
        elif args.mode == 'label+topic':
            per_group[label] += emb
            per_group_count[label] += 1
            per_group[topic] += emb
            per_group_count[topic] += 1

for example in tqdm.tqdm(examples, miniters=10000):
    while len(example_batch) >= 64:
        process_batch(args, device, tokenizer, model, per_group, per_group_count, example_batch[:64])

        example_batch = example_batch[64:]
    else:
       example_batch.append(example)

# process last few
process_batch(args, device, tokenizer, model, per_group, per_group_count, example_batch)

for group in per_group:
    per_group[group] /= per_group_count[group]

per_group = dict(per_group)

with open(f'{args.dataset}_per_{args.mode}_emb_test.pkl', 'wb') as f:
    pickle.dump(per_group, f)
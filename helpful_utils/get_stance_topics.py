import argparse
import json
from run_classify import PROCESSORS


parser = argparse.ArgumentParser()
parser.add_argument('--datasets', nargs='*')
parser.add_argument('--stance_ds_folder', default='/projects/tir5/users/jinglim/stance_datasets')

args = parser.parse_args()
args.task_name = 'stance'
topics = set()

for dataset in args.datasets:
    print(dataset)
    if dataset.startswith('comb_nlpcc_'):
        count = dataset.split('_')[-1]
        processor = PROCESSORS[args.task_name]['comb_nlpcc_fs'](count, args.seed)
    elif dataset.startswith('maj_twitter_bulk_emb-'):
        neighbours = int(dataset.split('-')[1])
        variant = dataset.split('-')[2]
        processor = PROCESSORS[args.task_name]['maj_twitter_bulk_emb'](neighbours, variant)
    elif dataset.startswith('maj_twitter_v2_bulk_emb-'):
        neighbours = int(dataset.split('-')[1])
        variant = dataset.split('-')[2]
        processor = PROCESSORS[args.task_name]['maj_twitter_v2_bulk_emb'](neighbours, variant, 2)
    elif dataset.startswith('maj_twitter_bulk_'):
        count = int(dataset.split('_')[-1])
        processor = PROCESSORS[args.task_name]['maj_twitter_bulk'](count, args.seed)
    elif dataset.startswith('maj_twitter_tfidf_'):
        count = int(dataset.split('_')[-1])
        processor = PROCESSORS[args.task_name]['maj_twitter_tfidf'](count, args.seed)
    elif dataset.startswith('nusax_'):
        nusa_lang = '_'.join(dataset.split('_')[1:])
        processor = PROCESSORS[args.task_name]['nusax'](nusa_lang)
    elif dataset.startswith('xstance_'):
        xstance_lang = dataset.split('_')[-1]
        processor = PROCESSORS[args.task_name]['xstance'](xstance_lang)
    elif dataset.startswith('zh_'):
        real_ds = dataset.split('_')[-1]
        processor = PROCESSORS[args.task_name]['zh'](real_ds)
    else:
        processor = PROCESSORS[args.task_name][dataset]()

    examples = processor.get_train_examples(args.stance_ds_folder)

    for example in examples:
        topics.add(example.topic)

with open('stance_ds_topics.json', 'w') as f:
    json.dump(list(topics), f, indent=0)
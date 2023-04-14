import argparse
import logging
import openai
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support
import time
import tqdm

from run_classify import get_processor

import openai_secrets

parser = argparse.ArgumentParser()
parser.add_argument('--train_dataset', required=True, help='train example')
parser.add_argument('--dataset', required=True, help='test dataset')
parser.add_argument('--data_dir', default='../stance_datasets')
parser.add_argument('--to_skip', type=int, default=0)

args = parser.parse_args()

logger = logging.getLogger(__name__)

nlpcc_label_map = {'AGAINST': 'against',
                   'FAVOR': 'in favour',
                   'NONE': 'neutral'}

openai.api_key = openai_secrets.secret_key

MODEL = 'gpt-3.5-turbo'

train_processor = get_processor(args.train_dataset, 'stance', seed=43)
train_examples = train_processor.get_train_examples(args.data_dir)
train_labels = train_processor.get_labels()
train_labels = [f"'{label}'" for label in train_labels]

processor = get_processor(args.dataset, 'stance', seed=43)
examples = processor.get_test_examples(args.data_dir)
labels = processor.get_labels()
labels = [f"'{label}'" for label in labels]

counter_name = {1: 'one',
                2: 'two',
                3: 'three',
                4: 'four',
                5: 'five'}

outputs = []

for i, example in enumerate(tqdm.tqdm(examples)):
    if i < args.to_skip:
        continue
    try:
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": f"You are a stance detection assistant which only replies with {', '.join(labels[:-1])} or {labels[-1]}."},
                {"role": "user", "content": f"Which of the {counter_name[len(train_labels)]} choices, {', '.join(train_labels)} is the closest to the stance of the author of '{train_examples[0].text}' towards '{train_examples[0].topic}'"},
                {"role": "assistant", "content": "in favour'"},
                {"role": "user", "content": f"Which of the {counter_name[len(labels)]} choices, {', '.join(labels)} is the closest to the stance of the author of '{example.text}' towards '{example.topic}'"},
            ],
            temperature=0,
        )
    except openai.error.RateLimitError:
        time.sleep(2)
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": f"You are a stance detection assistant which only replies with {', '.join(labels[:-1])} or {labels[-1]}."},
                {"role": "user", "content": f"Which of the {counter_name[len(labels)]} choices, {', '.join(labels)} is the closest to the stance of the author of '{examples[0].text}' towards '{examples[0].topic}'"},
                {"role": "assistant", "content": "in favour'"},
                {"role": "user", "content": f"Which of the {counter_name[len(labels)]} choices, {', '.join(labels)} is the closest to the stance of the author of '{example.text}' towards '{example.topic}'"},
            ],
            temperature=0,
        )
    try:
        outputs.append(response['choices'][0]['message']['content'])
    except (TypeError, IndexError) as e:
        logger.info(i)
        logger.info(e)
    time.sleep(0.6)
    if i % 100 == 0:
        with open(f'chatgpt_api_{args.dataset}_test_outputs.txt', 'w') as f:
            for o_id, result in enumerate(outputs):
                f.write(f'{o_id+args.to_skip},{result}\n')


with open(f'chatgpt_api_{args.dataset}_test_outputs.txt', 'w') as f:
    for o_id, result in enumerate(outputs):
        f.write(f'{o_id+args.to_skip},{result}\n')

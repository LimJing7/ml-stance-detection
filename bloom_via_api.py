import datasets
import json
import logging
import requests
from sklearn.metrics import f1_score, precision_recall_fscore_support
import tqdm

logger = logging.getLogger(__name__)

nlpcc_label_map = {'AGAINST': 'against',
                   'FAVOR': 'in favour',
                   'NONE': 'neutral'}


API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"

with open('/home/jinglim/.huggingface/token') as f:
    API_TOKEN = f.read()
headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

def construct_prompt(examples, number_to_use):
    prompt = ''
    for i, example in enumerate(examples):
        if i == number_to_use:
            break
        sent = f'Regarding {example["topic"]}, stance of {example["text"]} is {nlpcc_label_map[example["label"]]}; '
        prompt += sent
    return prompt

ds = datasets.load_dataset('limjing7/nlpcc', use_auth_token=True)
train_ds = ds['train']
dev_ds = ds['dev']

topics = ["IphoneSE", "春节放鞭炮", "深圳禁摩限电", "俄罗斯在叙利亚的反恐行动", "开放二胎"]
for topic in topics:
    filtered_train = train_ds.filter(lambda x: x['topic'] == topic).shuffle(seed=43)
    prompt = construct_prompt(filtered_train, 7)

    output_file =  open(f'bloom_{topic}.csv', 'w')

    preds = []
    golds = []

    for i, example in enumerate(tqdm.tqdm(dev_ds)):
        input_string = f'{prompt}Regarding {example["topic"]}, stance of {example["text"]} is'
        input_len = len(input_string)
        gold = nlpcc_label_map[example["label"]]
        data = query({"inputs": input_string})
        try:
            pred = data[0]['generated_text'][input_len:].split(';')[0].strip()
            output_file.write(f'{gold},{pred},"{example["text"]}",{example["topic"]}\n')

            golds.append(gold)
            preds.append(pred)
        except KeyError:
            print(i)
            print(input_string)
            print(data)

    output_file.close()

    with open(f'bloom_{topic}_results', 'w') as f:
        f.write(f'Prompt = {prompt}\n')
        labels_order = list(set(golds))
        precision, recall, f1, support = precision_recall_fscore_support(golds, preds, average=None, labels=labels_order)
        f.write(f'{labels_order = }\n')
        f.write(f'{precision = }\n')
        f.write(f'{recall = }\n')
        f.write(f'{f1 = }\n')


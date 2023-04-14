import argparse
import numpy as np
import pandas as pd
import torch
import tqdm
import transformers

parser = argparse.ArgumentParser()
parser.add_argument('--part_id', type=int)

args = parser.parse_args()

device = torch.device('cuda')
batch_size = 16

tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('microsoft/infoxlm-base')
model = transformers.XLMRobertaModel.from_pretrained('microsoft/infoxlm-base').to(device)
my_useful_model = torch.nn.Sequential(torch.nn.Linear(768,768), torch.nn.ReLU(), torch.nn.Linear(768,1)).to(device)
my_useful_model.load_state_dict(torch.load('./useful_model'))

# for part in tqdm.trange(12):
part = args.part_id
df = pd.read_csv(f'twitter_data/majority_voted_twitter_wiki_cont_part_{part}.csv',lineterminator='\n')

texts = []
for i, row in df.iterrows():
    if row.text is np.nan:
        texts.append('')
    else:
        texts.append(row.text)

ys = []
with torch.no_grad():
    for i in tqdm.trange(0, len(texts), batch_size):
        text = texts[i:i+batch_size]
        enc = tokenizer.batch_encode_plus(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        enc = {i: j.to(device) for i,j in enc.items()}
        output = model(**enc).pooler_output
        o = my_useful_model(output.detach()).detach().cpu()
        y = o > 0
        ys.append(y)
ys = torch.cat(ys)

df['classifier_pred'] = ys

df.to_csv(f'twitter_data/majority_voted_twitter_wiki_cont_class_filt_part_{part}.csv', index=False)


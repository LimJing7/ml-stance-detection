import argparse
import heapq
import os
from typing import Iterator
import pandas as pd
import pickle
import torch
from torch.utils.data import IterableDataset, DataLoader
import tqdm
import transformers

class NearestK:
    def __init__(self, center, n_points):
        self.center = center.reshape(1, -1)
        self.n_points = n_points
        self.list = []

    def add(self, value, obj):
        dist = torch.cdist(self.center, value.reshape(1, -1), p=2)[0,0]
        self.list.append((dist, obj))

    def sort(self):
        self.list = sorted(self.list)

    def truncate(self):
        '''
        Drop all the large items until only n_points items remain
        '''
        self.sort()
        self.list = self.list[:self.n_points]

    def add_multiple(self, values, objs):
        with torch.no_grad():
            dists = torch.cdist(self.center, values, p=2)[0].cpu().numpy()
        self.list += list(zip(dists, objs))

    def add_multiple_with_dist(self, dists, objs):
        self.list += list(zip(dists, objs))


class MyDataset(IterableDataset):
    def __init__(self, data1, data2):
        super(MyDataset).__init__()
        self.data1 = data1
        self.data2 = data2
    def __iter__(self):
        return iter(zip(self.data1, self.data2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--center_file')
    parser.add_argument('--n_points_per_center', type=int)
    parser.add_argument('--tweet_file')
    parser.add_argument('--model_file')
    parser.add_argument('--output_file')

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.batch_size = 16

    with open(args.center_file, 'rb') as f:
        centers = pickle.load(f)

    tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(args.model_file)
    model = transformers.XLMRobertaModel.from_pretrained(args.model_file).to(args.device)

    selected = {}
    centers_list = []
    for center in sorted(centers):
        selected[center] = NearestK(centers[center].detach().to(args.device), args.n_points_per_center)
        centers_list.append(centers[center].detach())

    centers_tensor = torch.stack(centers_list).to(args.device)

    df = pd.read_csv(args.tweet_file, lineterminator='\n')

    dataset = MyDataset(df['text'], df.index)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    for batch in tqdm.tqdm(dataloader, total=(df.shape[0]//args.batch_size)):
        text, idx = batch
        toked = tokenizer(text, return_tensors='pt', padding=True)
        embs = model(toked['input_ids'].to(args.device), toked['attention_mask'].to(args.device))['pooler_output'].detach()
        dists = torch.cdist(centers_tensor, embs, p=2)
        sel_centers = torch.min(dists, axis=0).indices
        for c_id, center in enumerate(sorted(centers)):
            sel_dists = dists[c_id, sel_centers == c_id]
            sel_idx = idx[sel_centers == c_id]
            selected[center].add_multiple_with_dist(sel_dists.cpu().numpy(), sel_idx.cpu().numpy())
            # selected[center].add_multiple(embs, idx.cpu().numpy())  ## this version has being deprecated
        del embs

    all_sel_idx = []
    for center in selected:
        selected[center].truncate()
        sel_idx = [i[1] for i in selected[center].list]
        all_sel_idx.extend(sel_idx)

    print(all_sel_idx)

    sel_df = df[df.index.isin(set(all_sel_idx))]
    sel_df.to_csv(args.output_file, index=False)

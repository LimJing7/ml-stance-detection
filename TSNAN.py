import argparse
from collections import defaultdict
from csv import QUOTE_ALL
import pandas as pd
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import tqdm
import transformers

import hanlp

class TSNAN(nn.Module):
    def __init__(self, n_emb, embed_size, lstm_size, output_size) -> None:
        super().__init__()

        self.embedding = torch.nn.Embedding(n_emb, embed_size)
        self.lstm = torch.nn.LSTM(input_size=embed_size, hidden_size=lstm_size, bidirectional=True, batch_first=True)

        self.target_linear = torch.nn.Linear(2*embed_size, 1)
        self.target_softmax = torch.nn.Softmax(dim=1)

        self.output_linear = torch.nn.Linear(2*lstm_size, output_size)

    def forward(self, topic, topic_mask, text, text_mask):
        topic_lens = torch.sum(topic_mask, axis=1, keepdim=True)  # batch * 1
        text_lens = torch.sum(text_mask, axis=1, keepdim=True)  # batch * 1

        text_emb = self.embedding(text)  # batch * len * emb
        text_emb_packed = pack_padded_sequence(text_emb, text_lens.squeeze(), batch_first=True, enforce_sorted=False)

        topic_emb = self.embedding(topic)  # batch * len * emb
        topic_emb = torch.mul(topic_emb, topic_mask.unsqueeze(2))
        topic_emb = torch.sum(topic_emb, axis=1, keepdim=True)  # batch * 1 * emb
        topic_emb = torch.div(topic_emb, topic_lens.unsqueeze(2)).expand(text_emb.shape)  # batch * 1 * emb

        lstm_out, states = self.lstm(text_emb_packed)
        lstm_out = pad_packed_sequence(lstm_out, batch_first=True, total_length=text_mask.shape[1])[0]

        cat_emb = torch.cat([text_emb, topic_emb], dim=2)
        attn = self.target_linear(cat_emb) # batch * len * 1
        attn = torch.mul(attn, text_mask.unsqueeze(2))
        attn = self.target_softmax(attn)

        comb = torch.mul(lstm_out, attn)
        sent_emb = torch.sum(comb, dim=1) # batch * 2lstm_size
        sent_emb = torch.div(sent_emb, text_lens)

        output = self.output_linear(sent_emb)

        return output


class NBOW(nn.Module):
    def __init__(self, n_emb, embed_size, lstm_size, output_size) -> None:
        super().__init__()

        self.embedding = torch.nn.Embedding(n_emb, embed_size)

        self.output_linear = torch.nn.Linear(embed_size, output_size)

    def forward(self, topic, topic_mask, text, text_mask):
        topic_lens = torch.sum(topic_mask, axis=1, keepdim=True)  # batch * 1
        text_lens = torch.sum(text_mask, axis=1, keepdim=True)  # batch * 1

        text_emb = self.embedding(text)  # batch * len * emb
        text_emb = torch.mul(text_emb, text_mask.unsqueeze(2))
        text_emb = torch.sum(text_emb, axis=1)  # batch * emb
        text_emb = torch.div(text_emb, text_lens)  # batch * emb

        output = self.output_linear(text_emb)

        return output



def convert_tokens_to_ids(text, tokens_dict, return_tensors=None):
    output = []
    mask = []
    max_length = 0
    for row in text:
        output_row = []
        for word in row:
            output_row.append(tokens_dict[word])
        if len(output_row) > max_length:
            max_length = len(output_row)
        output.append(output_row)
        mask.append([1]*len(output_row))

    if return_tensors == 'pt':
        output_tensor = torch.ones((len(output), max_length), dtype=int) * tokens_dict['pad']
        mask = torch.zeros((len(output), max_length), dtype=int)
        for i, row in enumerate(output):
            output_tensor[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        return output_tensor, mask

    return output, mask


def create_dataset(args, filename, tokenizer, label_map=None):
    df = pd.read_csv(filename, delimiter='\t', quotechar='"', header=None, quoting=QUOTE_ALL)

    if args.topic is not None:
        df = df[df[1] == args.topic]

    topics = list(df[1])
    text = list(df[2])
    if args.word2vec:
        tokenizer_model, tokens_dict = tokenizer
        toked_topics = tokenizer_model(topics)
        tokenized_topics, topics_mask = convert_tokens_to_ids(toked_topics, tokens_dict, return_tensors='pt')
        toked_text = tokenizer_model(text)
        tokenized_text, text_mask = convert_tokens_to_ids(toked_text, tokens_dict, return_tensors='pt')
    else:
        toked_topics = tokenizer.batch_encode_plus(topics, add_special_tokens=False, padding='longest', return_tensors='pt')
        tokenized_topics = toked_topics['input_ids']
        topics_mask = toked_topics['attention_mask']

        toked_text = tokenizer.batch_encode_plus(text, add_special_tokens=False, padding='longest', return_tensors='pt')
        tokenized_text = toked_text['input_ids']
        text_mask = toked_text['attention_mask']

    labels = list(df[3])
    if label_map is None:
        labels_set = sorted(list(set(labels)))
        label_map = {l:i for i,l in enumerate(labels_set)}
    labels = list(map(lambda x: label_map[x], labels))

    dataset = TensorDataset(tokenized_topics, topics_mask, tokenized_text, text_mask, torch.tensor(labels))
    return dataset, label_map


def train(args, model, tokenizer):
    tb_writer = SummaryWriter(f'runs/{args.output_dir.split("/")[-1]}')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_dataset, label_map = create_dataset(args, args.train_file, tokenizer)
    dev_dataset, _ = create_dataset(args, args.dev_file, tokenizer, label_map)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size)

    train_loss = []
    global_step = 0

    for epoch in tqdm.tqdm(range(args.epoch)):
        for i, batch in enumerate(tqdm.tqdm(train_dataloader)):
            topics, topics_mask, text, text_mask, labels = batch

            optimizer.zero_grad()
            outputs = model(topics, topics_mask, text, text_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if i%10:
                preds = torch.max(outputs, axis=1)[1]
                f1 = f1_score(labels, preds, average='macro')
                tb_writer.add_scalar('train loss', loss.item(), global_step)
                tb_writer.add_scalar('train f1', f1, global_step)

            global_step += 1

        dev_labels = torch.tensor([], dtype=int)
        dev_logits = torch.tensor([])
        dev_preds = torch.tensor([], dtype=int)
        for batch in dev_dataloader:
            topics, topics_mask, text, text_mask, labels = batch
            with torch.no_grad():
                outputs = model(topics, topics_mask, text, text_mask)
                preds = torch.max(outputs, axis=1)[1]

                dev_labels = torch.cat([dev_labels, labels])
                dev_logits = torch.cat([dev_logits, outputs], axis=0)
                dev_preds = torch.cat([dev_preds, preds])
        dev_loss = criterion(dev_logits, dev_labels)
        dev_f1 = f1_score(dev_labels, dev_preds, average="macro")

        print(f'preds: {dev_preds.unique(return_counts=True)}')
        print(f'labels: {dev_labels.unique(return_counts=True)}')
        print(f'dev f1: {f1_score(dev_labels, dev_preds, average=None)}')

        tb_writer.add_scalar('dev loss', dev_loss.item(), global_step)
        tb_writer.add_scalar('dev f1', dev_f1, global_step)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', default='./sgns.weibo.word', help='file containing word2vec vectors')
    parser.add_argument('--tokenizer', default='microsoft/infoxlm-base', help='tokenizer')
    parser.add_argument('--word2vec', action='store_true', help='use word2vec')
    parser.add_argument('--train_file')
    parser.add_argument('--dev_file')
    parser.add_argument('--output_dir', help='location to save trained model')
    parser.add_argument('--tsnan', action='store_true', help='use tsnan model')
    parser.add_argument('--nbow', action='store_true', help='use nbow model')
    parser.add_argument('--lstm_dim', default=100, type=int)
    parser.add_argument('--output_dim', default=3, type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--batch_size', default=50, type=int)
    parser.add_argument('--topic', default=None)
    parser.add_argument('--seed', default=43, type=int)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if args.word2vec:
        tokens = []

        with open(args.embedding_file) as f:
            header = f.readline()
            n_emb, embed_dim = map(int, header.strip().split())

            embedding_layer_weight = torch.zeros(n_emb, embed_dim)

            for i, line in enumerate(f):
                line = line.split()
                token = line[0]
                tokens.append(token)
                emb = list(map(float, line[1:]))
                embedding_layer_weight[i] = torch.tensor(emb)

        unk_id = tokens.index('unknown')
        tokens_dict = defaultdict(lambda: unk_id)
        for i, token in enumerate(tokens):
            tokens_dict[token] = i

        tokenizer_model = hanlp.load(hanlp.pretrained.tok.CTB6_CONVSEG)
        tokenizer = (tokenizer_model, tokens_dict)

    else:

        tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(args.tokenizer)
        plm_model = transformers.XLMRobertaModel.from_pretrained(args.tokenizer)
        embedding_layer_weight = plm_model.embeddings.word_embeddings.weight

        n_emb, embed_dim = embedding_layer_weight.shape

    if args.tsnan:
        model = TSNAN(n_emb, embed_dim, args.lstm_dim, args.output_dim)
    elif args.nbow:
        model = NBOW(n_emb, embed_dim, args.lstm_dim, args.output_dim)

    if args.word2vec:
        model.embedding = torch.nn.Embedding.from_pretrained(embedding_layer_weight, freeze=False, padding_idx=tokens_dict['pad'])
    else:
        model.embedding = torch.nn.Embedding.from_pretrained(embedding_layer_weight, freeze=False, padding_idx=tokenizer.pad_token_id)

    train(args, model, tokenizer)



if __name__ == '__main__':
    main()
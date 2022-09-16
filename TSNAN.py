import argparse
from collections import defaultdict
from csv import QUOTE_ALL
import logging
import os
import pandas as pd
import pickle
from sklearn.metrics import f1_score, precision_recall_fscore_support
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import tqdm
import transformers

from run_classify import PROCESSORS

import hanlp

logger = logging.getLogger(__name__)

class TSNAN(nn.Module):
    def __init__(self, n_emb, embed_size, lstm_size, output_size, dropout_rate) -> None:
        super().__init__()

        self.embedding = torch.nn.Embedding(n_emb, embed_size)
        self.lstm = torch.nn.LSTM(input_size=embed_size, hidden_size=lstm_size, bidirectional=True, batch_first=True)

        self.target_linear = torch.nn.Linear(2*embed_size, 1)
        self.target_softmax = torch.nn.Softmax(dim=1)

        self.output_linear = torch.nn.Linear(2*lstm_size, output_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # self.apply(self._init_weights)

    def forward(self, topic, topic_mask, text, text_mask):
        topic_lens = torch.sum(topic_mask, axis=1, keepdim=True)  # batch * 1
        text_lens = torch.sum(text_mask, axis=1, keepdim=True)  # batch * 1

        text_emb = self.embedding(text)  # batch * len * emb
        text_emb_packed = pack_padded_sequence(text_emb, text_lens.squeeze().cpu(), batch_first=True, enforce_sorted=False)

        topic_emb = self.embedding(topic)  # batch * len * emb
        topic_emb = torch.mul(topic_emb, topic_mask.unsqueeze(2))
        topic_emb = torch.sum(topic_emb, axis=1, keepdim=True)  # batch * 1 * emb
        topic_emb = torch.div(topic_emb, topic_lens.unsqueeze(2)).expand(text_emb.shape)  # batch * 1 * emb

        lstm_out, states = self.lstm(text_emb_packed)
        lstm_out = pad_packed_sequence(lstm_out, batch_first=True, total_length=text_mask.shape[1])[0]
        lstm_out = self.dropout(lstm_out)

        cat_emb = torch.cat([text_emb, topic_emb], dim=2)
        attn = self.target_linear(cat_emb)  # batch * len * 1
        mask = torch.zeros_like(text_mask, dtype=torch.float32)
        mask[~text_mask.bool()] = float('-inf')
        attn = attn + mask.unsqueeze(2)
        # attn = torch.mul(attn, text_mask.unsqueeze(2))  # mask out padding
        attn = self.target_softmax(attn)

        comb = torch.mul(lstm_out, attn)
        sent_emb = torch.sum(comb, dim=1)  # batch * 2lstm_size
        sent_emb = torch.div(sent_emb, text_lens)
        sent_emb = self.dropout(sent_emb)

        output = self.output_linear(sent_emb)

        return output

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Embedding):
            torch.nn.init.uniform_(module.weight, -0.01, 0.01)
        elif isinstance(module, torch.nn.LSTM):
            for weight in module.parameters():
                torch.nn.init.uniform_(weight, -0.01, 0.01)
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.uniform_(module.weight, -0.01, 0.01)
            if module.bias is not None:
                torch.nn.init.uniform_(module.weight, -0.01, 0.01)


class NBOW(nn.Module):
    def __init__(self, n_emb, embed_size, lstm_size, output_size, dropout_rate) -> None:
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

def create_dataset(args, dataset_name, tokenizer, label_map=None, split='train'):
    processor = PROCESSORS['stance'][dataset_name]()
    labels = processor.get_labels()

    if split == 'train':
        examples = processor.get_train_examples(args.data_dir)
    elif split == 'translate-train':
        examples = processor.get_translate_train_examples(args.data_dir)
    elif split == 'translate-test':
        examples = processor.get_translate_test_examples(args.data_dir)
    elif split == 'dev':
        examples = processor.get_dev_examples(args.data_dir)
    elif split == 'pseudo_test':
        examples = processor.get_pseudo_test_examples(args.data_dir)
    else:
        examples = processor.get_test_examples(args.data_dir)

    if args.topic is not None and args.topic != 'all':
        examples = [example for example in examples if example.topic == args.topic]
        logger.info(f'{args.topic=}')
        logger.info(f'{len(examples)=}')

    topics = [example.topic for example in examples]
    text = [example.text for example in examples]
    labels = [example.label for example in examples]

    if args.word2vec:
        tokenizer_model, tokens_dict = tokenizer
        if args.en:
            toked_topics = [[i.lower_ for i in tokenizer_model(topic)] for topic in topics]
            toked_text = [[i.lower_ for i in tokenizer_model(txt)] for txt in text]
        else:
            toked_topics = tokenizer_model(topics)
            toked_text = tokenizer_model(text)
        tokenized_topics, topics_mask = convert_tokens_to_ids(toked_topics, tokens_dict, return_tensors='pt')
        tokenized_text, text_mask = convert_tokens_to_ids(toked_text, tokens_dict, return_tensors='pt')
    else:
        toked_topics = tokenizer.batch_encode_plus(topics, add_special_tokens=False, padding='longest', return_tensors='pt')
        tokenized_topics = toked_topics['input_ids']
        topics_mask = toked_topics['attention_mask']

        toked_text = tokenizer.batch_encode_plus(text, add_special_tokens=False, padding='longest', return_tensors='pt')
        tokenized_text = toked_text['input_ids']
        text_mask = toked_text['attention_mask']

    if label_map is None:
        labels_set = processor.get_labels()
        label_map = {l:i for i,l in enumerate(labels_set)}
    labels = list(map(lambda x: label_map[x], labels))

    dataset = TensorDataset(tokenized_topics, topics_mask, tokenized_text, text_mask, torch.tensor(labels))
    return dataset, label_map


def create_dataset_by_filename(args, filename, tokenizer, label_map=None):
    df = pd.read_csv(filename, delimiter='\t', quotechar='"', header=None, quoting=QUOTE_ALL)

    if args.topic is not None and args.topic != 'all':
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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_lambda)

    if args.train_dataset is not None:
        train_dataset, label_map = create_dataset(args, args.train_dataset, tokenizer, split='train')
    else:
        train_dataset, label_map = create_dataset_by_filename(args, args.train_file, tokenizer)

    if args.dev_dataset is not None:
        dev_dataset, _ = create_dataset(args, args.dev_dataset, tokenizer, split='dev', label_map=label_map)
    else:
        dev_dataset, _ = create_dataset_by_filename(args, args.dev_file, tokenizer, label_map)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.per_gpu_train_batch_size)

    dev_dataloader = DataLoader(dev_dataset, batch_size=args.per_gpu_train_batch_size)

    n_train_examples = len(train_dataloader)
    t_total = n_train_examples * args.num_train_epochs

    train_loss = []
    global_step = 0

    best_score = 0
    best_checkpoint = None

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", n_train_examples)
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    for epoch in tqdm.tqdm(range(args.num_train_epochs)):
        for i, batch in enumerate(tqdm.tqdm(train_dataloader)):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            topics, topics_mask, text, text_mask, labels = batch

            optimizer.zero_grad()
            outputs = model(topics, topics_mask, text, text_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if i%10:
                preds = torch.max(outputs, axis=1)[1]
                f1 = f1_score(labels.cpu(), preds.detach().cpu(), average='macro')
                tb_writer.add_scalar('train loss', loss.item(), global_step)
                tb_writer.add_scalar('train f1', f1, global_step)

            global_step += 1

        model.eval()
        dev_labels = torch.tensor([], dtype=int)
        dev_logits = torch.tensor([])
        dev_preds = torch.tensor([], dtype=int)
        for batch in dev_dataloader:
            batch = tuple(t.to(args.device) for t in batch)
            topics, topics_mask, text, text_mask, labels = batch
            with torch.no_grad():
                outputs = model(topics, topics_mask, text, text_mask)
                preds = torch.max(outputs, axis=1)[1]

                dev_labels = torch.cat([dev_labels, labels.cpu()])
                dev_logits = torch.cat([dev_logits, outputs.detach().cpu()], axis=0)
                dev_preds = torch.cat([dev_preds, preds.detach().cpu()])
        dev_loss = criterion(dev_logits, dev_labels)
        dev_f1 = f1_score(dev_labels, dev_preds, average="macro")

        logger.info(f'preds: {dev_preds.unique(return_counts=True)}')
        logger.info(f'labels: {dev_labels.unique(return_counts=True)}')
        logger.info(f'dev f1: {f1_score(dev_labels, dev_preds, average=None)}')

        tb_writer.add_scalar('dev loss', dev_loss.item(), global_step)
        tb_writer.add_scalar('dev f1', dev_f1, global_step)

        if dev_f1 > best_score:
            logger.info(" result['macro f1']={} > best_score={}".format(dev_f1, best_score))
            output_dir = os.path.join(args.output_dir, "checkpoint-best")
            best_checkpoint = output_dir
            best_score = dev_f1
            # Save model checkpoint
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "model.pt"))
            if args.word2vec:
                tokenizer_model = tokenizer[0]
                tokenizer_model.save(os.path.join(output_dir, "tokenizer"))
                tokenizer_dict = dict(tokenizer[1])
                unk_id = tokenizer_dict['unknown']
                pickle.dump((tokenizer_dict, unk_id), open(os.path.join(output_dir, "tokenizer", "tokens_dict.pkl"), 'wb'))
            else:
                tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)

            pickle.dump(label_map, open(os.path.join(output_dir, "label_map.pkl"), 'wb'))
            logger.info("Saving label map to %s", output_dir)

    return best_score, best_checkpoint

def evaluate(args, model, tokenizer, label_map):
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    if args.dev_dataset is not None:
        dev_dataset, _ = create_dataset(args, args.dev_dataset, tokenizer, split='dev', label_map=label_map)
    else:
        dev_dataset, _ = create_dataset_by_filename(args, args.dev_file, tokenizer, label_map)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.per_gpu_train_batch_size)

    results = {}

    dev_labels = torch.tensor([], dtype=int)
    dev_logits = torch.tensor([])
    dev_preds = torch.tensor([], dtype=int)
    for batch in dev_dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        topics, topics_mask, text, text_mask, labels = batch
        with torch.no_grad():
            outputs = model(topics, topics_mask, text, text_mask)
            preds = torch.max(outputs, axis=1)[1]

            dev_labels = torch.cat([dev_labels, labels.cpu()])
            dev_logits = torch.cat([dev_logits, outputs.detach().cpu()], axis=0)
            dev_preds = torch.cat([dev_preds, preds.detach().cpu()])
    dev_loss = criterion(dev_logits, dev_labels)
    macro_dev_f1 = f1_score(dev_labels, dev_preds, average="macro")
    preds_count = dev_preds.unique(return_counts=True)
    labels_count = dev_labels.unique(return_counts=True)
    dev_per_cls_f1 = f1_score(dev_labels, dev_preds, average=None)
    precision, recall, f1, support = precision_recall_fscore_support(dev_labels, dev_preds, average=None)

    logger.info(f'preds: {preds_count}')
    logger.info(f'labels: {labels_count}')
    logger.info(f'dev f1: {dev_per_cls_f1}')
    logger.info(f'{dev_loss=}')
    logger.info(f'{macro_dev_f1=}')

    results['preds_count'] = preds_count
    results['labels_count'] = labels_count
    results['macro_dev_f1'] = macro_dev_f1
    results['dev loss'] = dev_loss
    results[f'precisions'] = precision
    results[f'recalls'] = recall
    results['dev f1'] = dev_per_cls_f1
    results['label_map'] = label_map

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', default='./sgns.weibo.word', help='file containing word2vec vectors')
    parser.add_argument('--tokenizer', default='microsoft/infoxlm-base', help='tokenizer')
    parser.add_argument('--word2vec', action='store_true', help='use word2vec')
    parser.add_argument('--en', action='store_true', help='use en mode')
    parser.add_argument('--do_train', action='store_true', help='train')
    parser.add_argument('--do_eval', action='store_true', help='evaluate')
    parser.add_argument('--train_file')
    parser.add_argument('--train_dataset')
    parser.add_argument('--dev_file')
    parser.add_argument('--dev_dataset')
    parser.add_argument('--data_dir', help='directory to look for datasets')
    parser.add_argument('--output_dir', help='location to save trained model')
    parser.add_argument('--tsnan', action='store_true', help='use tsnan model')
    parser.add_argument('--nbow', action='store_true', help='use nbow model')
    parser.add_argument('--lstm_dim', default=100, type=int)
    parser.add_argument('--output_dim', default=3, type=int)
    parser.add_argument('--learning_rate', default=5e-4, type=float)
    parser.add_argument('--l2_lambda', default=0, type=float)
    parser.add_argument('--dropout_rate', default=0, type=float)
    parser.add_argument('--num_train_epochs', default=10, type=int)
    parser.add_argument('--per_gpu_train_batch_size', default=50, type=int)
    parser.add_argument('--topic', default=None)
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--log_file", default="train", type=str, help="log file")
    parser.add_argument('--seed', default=43, type=int)
    parser.add_argument('--no_cuda', action='store_true', help='set if no cuda wanted')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logging.basicConfig(handlers=[logging.FileHandler(os.path.join(args.output_dir, args.log_file)), logging.StreamHandler()],
                      format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                      datefmt='%m/%d/%Y %H:%M:%S',
                      level=logging.INFO)
    logger.info(args)

    torch.manual_seed(args.seed)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if args.word2vec:
        tokens = []

        logger.info('loading embeddings')
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

        if args.en:
            nlp = spacy.lang.en.English()
            tokenizer = nlp.tokenizer
            tokenizer = (tokenizer_model, tokens_dict)
        else:
            tokenizer_model = hanlp.load(hanlp.pretrained.tok.CTB6_CONVSEG)
            tokenizer = (tokenizer_model, tokens_dict)

    else:

        tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(args.tokenizer)
        plm_model = transformers.XLMRobertaModel.from_pretrained(args.tokenizer)
        embedding_layer_weight = plm_model.embeddings.word_embeddings.weight

        n_emb, embed_dim = embedding_layer_weight.shape

    if args.tsnan:
        model = TSNAN(n_emb, embed_dim, args.lstm_dim, args.output_dim, args.dropout_rate)
    elif args.nbow:
        model = NBOW(n_emb, embed_dim, args.lstm_dim, args.output_dim)
    else:
        raise KeyError('Choose either tsnan or nbow')

    if args.word2vec:
        model.embedding = torch.nn.Embedding.from_pretrained(embedding_layer_weight, freeze=False, padding_idx=tokens_dict['pad'])
    else:
        model.embedding = torch.nn.Embedding.from_pretrained(embedding_layer_weight, freeze=False, padding_idx=tokenizer.pad_token_id)

    logging.info('move model to gpu')
    model = model.to(args.device)
    logging.info('model moved to gpu')

    if args.do_train:
        best_score, best_checkpoint = train(args, model, tokenizer)
        logger.info("best checkpoint = {}, best score = {}".format(best_checkpoint, best_score))

    if args.do_eval:
        logger.info('Do eval')
        if args.word2vec:
            # tokenizer_model = hanlp.load(os.path.join(args.output_dir, "tokenizer"))
            tokenizer_model = hanlp.load(hanlp.pretrained.tok.CTB6_CONVSEG)
            tokenizer_dict, unk_id = pickle.load(open(os.path.join(args.output_dir, "tokenizer", "tokens_dict.pkl"), 'rb'))
            tokens_dict = defaultdict(lambda: unk_id, tokenizer_dict)
            tokenizer = (tokenizer_model, tokens_dict)
        else:
            tokenizer = transformers.XLMRobertaTokenizer.from_pretrained(args.output_dir)

        checkpoints = [args.output_dir]
        best_checkpoint = None
        best_score = 0
        for checkpoint in checkpoints:
            model.load_state_dict(torch.load(os.path.join(args.output_dir, "model.pt")))
            label_map = pickle.load(open(os.path.join(args.output_dir, "label_map.pkl"), 'rb'))
            results = evaluate(args, model, tokenizer, label_map)
            output_eval_file = os.path.join(args.output_dir, 'eval_results')

            if results['macro_dev_f1'] > best_score:
                best_checkpoint = checkpoint
                best_score = results['macro_dev_f1']

        with open(output_eval_file, 'w') as writer:
            for key, value in results.items():
                writer.write('{} = {}\n'.format(key, value))
            writer.write("Best checkpoint is {}, best accuracy is {}".format(best_checkpoint, best_score))
            logger.info("Best checkpoint is {}, best accuracy is {}".format(best_checkpoint, best_score))


if __name__ == '__main__':
    main()
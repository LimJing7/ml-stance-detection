# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors,
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning multi-lingual models on stance detection."""


import argparse
import glob
import logging
import os
import pickle
import random
import statistics

import numpy as np
import sklearn
from sklearn.metrics import f1_score
import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
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
  XLMRobertaForMaskedLM,
  get_linear_schedule_with_warmup,
)
import transformers
from processors.amazonzh import AmazonZhProcessor
from processors.idclickbait import IdClickbaitProcessor
from processors.nli_for_simcse import NLIforSIMCSEProcessor
from processors.parallel_nli import ParallelNLIProcessor
from processors.pawsen import PAWSXEnProcessor
from processors.pawszh import PAWSXZhProcessor
from processors.twitter2015 import Twitter2015Processor
from processors.trans_nli_for_simcse import TransNLIforSIMCSEProcessor
from processors.webtext2019 import Webtext2019Processor

from processors.utils import convert_examples_to_stance_features, convert_examples_to_parallel_features, convert_examples_to_mlm_features

from processors.ans import ANSProcessor
from processors.argmin import ArgMinProcessor
from processors.arc import ARCProcessor
from processors.fnc1 import FNC1Processor
from processors.iac1 import IAC1Processor
from processors.ibmcs import IBMCSProcessor
from processors.nlpcc import NLPCCProcessor
from processors.perspectrum import PerspectrumProcessor
from processors.semeval2016t6 import SemEval2016t6Processor
from processors.snopes import SnopesProcessor
from processors.trans_nlpcc import TransNLPCCProcessor
from processors.twitter2015 import Twitter2015Processor
from processors.twitter2017 import Twitter2017Processor
from processors.vast import VASTProcessor
from processors.tnlpcc import tNLPCCProcessor

from processors.indonli import IndonliProcessor

from perturb import perturb

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
  "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
  "xlm": (XLMConfig, XLMWithLMHeadModel, XLMTokenizer),
  "xlmr": (XLMRobertaConfig, XLMRobertaForMaskedLM, XLMRobertaTokenizer),
}

PROCESSORS = {
  'stance': {'ans': ANSProcessor,
             'arc': ARCProcessor,
             'argmin': ArgMinProcessor,
             'fnc1': FNC1Processor,
             'iac1': IAC1Processor,
             'ibmcs': IBMCSProcessor,
             'nlpcc': NLPCCProcessor,
             'perspectrum': PerspectrumProcessor,
             'semeval2016t6': SemEval2016t6Processor,
             'snopes': SnopesProcessor,
             'trans_nlpcc': TransNLPCCProcessor,
             'twitter2015': Twitter2015Processor,
             'twitter2017': Twitter2017Processor,
             'vast': VASTProcessor,
             'tnlpcc': tNLPCCProcessor},
  'no_topic_stance': {'ans': ANSProcessor,
                      'arc': ARCProcessor,
                      'argmin': ArgMinProcessor,
                      'fnc1': FNC1Processor,
                      'iac1': IAC1Processor,
                      'ibmcs': IBMCSProcessor,
                      'nlpcc': NLPCCProcessor,
                      'perspectrum': PerspectrumProcessor,
                      'semeval2016t6': SemEval2016t6Processor,
                      'snopes': SnopesProcessor,
                      'trans_nlpcc': TransNLPCCProcessor,
                      'twitter2015': Twitter2015Processor,
                      'twitter2017': Twitter2017Processor,
                      'vast': VASTProcessor,
                      'tnlpcc': tNLPCCProcessor},
  'nli': {'indonli': IndonliProcessor},
  'classification': {'amazonzh': AmazonZhProcessor,
                     'idclickbait': IdClickbaitProcessor},
  'pawsx': {'pawsxen': PAWSXEnProcessor,
            'pawsxzh': PAWSXZhProcessor},
  'parallel': {'parallel_nli': ParallelNLIProcessor,
               'nli_for_simcse': NLIforSIMCSEProcessor,
               'trans_nli_for_simcse': TransNLIforSIMCSEProcessor},
  'mlm': {'webtext2019': Webtext2019Processor},
}


def get_compute_preds(args, tokenizer, model, datasets):
  if type(datasets) != list:
    datasets = [datasets]
  embeded_tokens = []
  for ds in datasets:
    processor = PROCESSORS[args.task_name][ds]()
    labels = processor.get_labels()
    for label in labels:
      lab_embed = torch.mean(model.roberta.embeddings(tokenizer.encode(label, add_special_tokens=False, return_tensors='pt').to(model.device))[0], axis=0)
      embeded_tokens.append(lab_embed)
  LE = torch.stack(embeded_tokens).detach()
  def compute_preds(preds, shifts, ends):
    output = torch.zeros(preds.shape[0], dtype=torch.long)
    scores = (preds @ LE.T)
    for i, (score, shift, end) in enumerate(zip(scores, shifts, ends)):
      output[i] = torch.max(score[shift:end], axis=0)[1] + shift
    return output
  return compute_preds

def get_compute_loss(args, tokenizer, model, datasets):
  toked_labels = []
  if type(datasets) != list:
    datasets = [datasets]
  for ds in datasets:
    processor = PROCESSORS[args.task_name][ds]()
    labels = processor.get_labels()
    for label in labels:
      lab_tok = tokenizer.encode(label, add_special_tokens=False, return_tensors='pt').to(model.device)
      toked_labels.append(lab_tok)

  if args.negative_samples > 0:
    negative_labels = {}
    for ds in datasets:
      processor = PROCESSORS[args.task_name][ds]()
      labels = processor.get_labels()
      negative_labels[ds] = {lab:set() for lab in labels}
      for label, syns in args.synonyms[ds].items():
        for key in negative_labels[ds].keys():
          if key != label:
            toked = [tokenizer.encode(i, add_special_tokens=False, return_tensors='pt').to(model.device) for i in syns]
            negative_labels[ds][key].update(toked)
    sorted_negative_labels = []
    for ds in datasets:
      processor = PROCESSORS[args.task_name][ds]()
      labels = processor.get_labels()
      for label in labels:
        sorted_negative_labels.append(list(negative_labels[ds][label]))

  if args.loss_fn == 'cross_entropy':
    def compute_loss(model, preds, labels, shifts, ends):
      embeded_labels = []
      for label in toked_labels:
        embed = torch.mean(model.roberta.embeddings(label)[0], axis=0)
        embeded_labels.append(embed)
      LE = torch.stack(embeded_labels).detach()

      scores = (preds @ LE.T).permute(0, 2, 1)
      loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(scores, labels)

      if args.negative_samples > 0:
        raise NotImplementedError
      return loss
  elif args.loss_fn == 'bce':
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    def compute_loss(model, preds, labels, shifts, ends, weights=None):
      embeded_labels = []
      for label in toked_labels:
        embed = torch.mean(model.roberta.embeddings(label)[0], axis=0)
        embeded_labels.append(embed)
      LE = torch.stack(embeded_labels).detach()
      n_labels = len(embeded_labels)

      scores = (preds @ LE.T).permute(0, 2, 1)
      pos = labels != -100
      labels = labels[pos]
      pos = pos.unsqueeze(1)
      pos = pos.expand(-1,n_labels,-1)
      scores = scores[pos].reshape(-1, n_labels)

      loss = 0
      for i in range(n_labels):
        # positive label
        idx = labels == i
        sc = scores[:, i][idx]
        if sc.shape[0] >= 1:
          p_loss = bce_loss(sc, torch.ones_like(sc))
          if args.ds_weights != 'equal' and weights is not None:
            p_loss *= weights[idx]
          loss += torch.mean(p_loss)

          # negative sampling
          if args.negative_samples > 0:
            n_syns = min(args.negative_samples, len(sorted_negative_labels[i]))
            if n_syns > 0:
              neg_samples = random.sample(sorted_negative_labels[i], k=n_syns)
              embeded_negs = []
              for neg_label in neg_samples:
                embed_neg = torch.mean(model.roberta.embeddings(neg_label)[0], axis=0)
                embeded_negs.append(embed_neg)
              neg_LE = torch.stack(embeded_negs).detach()
              neg_scores = (preds @ neg_LE.T).permute(0, 2, 1)
              neg_pos = pos[:, 0:1, :].expand(-1, n_syns, -1)
              neg_scores = neg_scores[neg_pos].reshape(-1, n_syns)
              for i_syns in range(n_syns):
                sc = neg_scores[:, i_syns][idx]
                ns_loss = bce_loss(sc, torch.zeros_like(sc))
                if args.ds_weights != 'equal' and weights is not None:
                  ns_loss *= weights[idx]
                loss += torch.mean(ns_loss)

        # negative labels
        idx_n = labels != i
        idx_n = torch.logical_and(idx_n, shifts <= i)
        idx_n = torch.logical_and(idx_n, ends > i)
        sc = scores[:, i][idx_n]
        if sc.shape[0] >= 1:
          n_loss = bce_loss(sc, torch.zeros_like(sc))
          if args.ds_weights != 'equal' and weights is not None:
            n_loss *= weights[idx_n]
          loss += torch.mean(n_loss)

      return loss

  return compute_loss


def compute_metrics(preds, labels):
  scores = {
    "acc": (preds == labels).mean(),
    "num": len(
      preds),
    "correct": (preds == labels).sum(),
    "macro f1": f1_score(labels, preds, average='macro')
  }
  return scores


def compute_ds_weights(args, datasets):
  sizes = []
  for ds in datasets:
    sizes.append(len(ds))
  p = sizes/np.sum(sizes)
  q = p**args.ds_alpha / np.sum(p**args.ds_alpha)
  if args.ds_weights == 'uncorrected_scaled':
    return q
  elif args.ds_weights == 'scaled':
    return q * len(sizes)
  elif args.ds_weights == 'inverse_scaled':
    return 1 / (q * len(sizes))
  elif args.ds_weights == 'random':
    un_normed = np.random.normal(scale=0.1, size=len(sizes))
    q = torch.softmax(torch.tensor(un_normed), dim=0).numpy()
    return q * len(sizes)
  else:
    raise NotImplementedError

def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, parallel_dataset, mlm_dataset, model, tokenizer, lang2id=None):
  """Train the model."""
  if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter(f'runs/{args.output_dir.split("/")[-1]}')

  datasets = args.train_dataset
  compute_loss = get_compute_loss(args, tokenizer, model, datasets)

  args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

  if args.sup_simcse:
    parallel_sampler = RandomSampler(parallel_dataset) if args.local_rank == -1 else DistributedSampler(parallel_dataset)
    parallel_dataloader = DataLoader(parallel_dataset, sampler=parallel_sampler, batch_size=args.train_batch_size)
    parallel_iterator = iter(parallel_dataloader)

  if args.extra_mlm:
    mlm_sampler = RandomSampler(mlm_dataset) if args.local_rank == -1 else DistributedSampler(mlm_dataset)
    mlm_dataloader = DataLoader(mlm_dataset, sampler=mlm_sampler, batch_size=args.train_batch_size)
    mlm_iterator = iter(mlm_dataloader)

  if args.max_steps > 0:
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
  else:
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

  # Prepare optimizer and schedule (linear warmup and decay)
  no_decay = ["bias", "LayerNorm.weight"]
  if args.model_type == 'xlmr':
    optimizer_grouped_parameters = [
      {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": args.weight_decay,
      },
      {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
      optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
  else:
    raise NotImplementedError

  # Check if saved optimizer or scheduler states exist
  if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    os.path.join(args.model_name_or_path, "scheduler.pt")
  ):
    # Load in optimizer and scheduler states
    optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

  if args.fp16:
    try:
      from apex import amp
    except ImportError:
      raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

  # multi-gpu training (should be after apex fp16 initialization)
  if args.n_gpu > 1:
    model = torch.nn.DataParallel(model)

  # Distributed training (should be after apex fp16 initialization)
  if args.local_rank != -1:
    model = torch.nn.parallel.DistributedDataParallel(
      model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    )

  # Train!
  logger.info("***** Running training *****")
  logger.info("  Num examples = %d", len(train_dataset))
  logger.info("  Num Epochs = %d", args.num_train_epochs)
  logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
  logger.info(
    "  Total train batch size (w. parallel, distributed & accumulation) = %d",
    args.train_batch_size
    * args.gradient_accumulation_steps
    * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
  )
  logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
  logger.info("  Total optimization steps = %d", t_total)

  global_step = 0
  epochs_trained = 0
  steps_trained_in_current_epoch = 0
  # Check if continuing training from a checkpoint
  if os.path.exists(args.model_name_or_path):
    # set global_step to gobal_step of last saved checkpoint from model path
    global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
    epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
    steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

    logger.info("  Continuing training from checkpoint, will skip to saved global_step")
    logger.info("  Continuing training from epoch %d", epochs_trained)
    logger.info("  Continuing training from global step %d", global_step)
    logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

  best_score = 0
  best_checkpoint = None
  tr_loss, logging_loss = 0.0, 0.0

  if args.nonsup_simcse or args.sup_simcse:
    cos_sim_fn = torch.nn.CosineSimilarity(dim=-1)
    simcse_loss_fn = torch.nn.CrossEntropyLoss()

  model.zero_grad()
  train_iterator = trange(
    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0], dynamic_ncols=True
  )
  set_seed(args)  # Added here for reproductibility
  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0], dynamic_ncols=True)
    for step, batch in enumerate(epoch_iterator):
      # Skip past any already trained steps if resuming training
      if steps_trained_in_current_epoch > 0:
        steps_trained_in_current_epoch -= 1
        continue

      model.train()
      batch = tuple(t.to(args.device) for t in batch)
      inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
      all_shifts = batch[4]
      all_ends = batch[5]
      if args.mlm:
        inputs['labels'] = batch[6]
      if args.model_type != "distilbert":
        inputs["token_type_ids"] = (
          batch[2] if args.model_type in ["bert"] else None
        )  # XLM don't use segment_ids
      if args.model_type == "xlm":
        if args.mlm:
          inputs["langs"] = batch[7]
        else:
          inputs["langs"] = batch[6]
      outputs = model(**inputs, output_hidden_states=True)

      if args.robust == 'rs_rp':
        loss = 0
        for _ in range(args.robust_samples):
          delta = (torch.rand_like(outputs['hidden_states'][-1]) - 0.5) * 2 * args.robust_size
          if args.ds_weights != 'equal':
            ds_weights = batch[-1]
            loss += compute_loss(model, outputs['hidden_states'][-1] + delta, batch[3], all_shifts, all_ends, ds_weights)
          else:
            loss += compute_loss(model, outputs['hidden_states'][-1] + delta, batch[3], all_shifts, all_ends)
        loss /= args.robust_samples
      else:
        if args.ds_weights != 'equal':
          ds_weights = batch[-1]
          loss = compute_loss(model, outputs['hidden_states'][-1], batch[3], all_shifts, all_ends, ds_weights)
        else:
          loss = compute_loss(model, outputs['hidden_states'][-1], batch[3], all_shifts, all_ends)

      if args.mlm:
        mlm_loss = outputs['loss']
        if args.ds_weights != 'equal':
          loss = args.alpha * loss + (1-args.alpha) * mlm_loss * batch[-1][0]  # not obv how to scale so just take the first one
        else:
          loss = args.alpha * loss + (1-args.alpha) * mlm_loss

      if args.nonsup_simcse:
        outputs2 = model(**inputs, output_hidden_states=True)
        v1 = outputs['hidden_states'][-1][:,0]
        v2 = outputs2['hidden_states'][-1][:,0]

        cos_sim = cos_sim_fn(v1.unsqueeze(1), v2.unsqueeze(0)) / args.simcse_temp
        simcse_labels = torch.arange(cos_sim.size(0)).long().to(args.device)
        simcse_loss = simcse_loss_fn(cos_sim, simcse_labels)

        loss = args.alpha * loss + (1-args.alpha) * simcse_loss

      if args.sup_simcse:
        try:
          parallel_batch = next(parallel_iterator)
        except StopIteration:
          parallel_iterator = iter(parallel_dataloader)
          parallel_batch = next(parallel_iterator)
        parallel_inputs = {"input_ids": parallel_batch[0].to(args.device), "attention_mask": parallel_batch[1].to(args.device)}

        inputs1 = {k:v[:, 0] for k,v in parallel_inputs.items()}
        inputs2 = {k:v[:, 1] for k,v in parallel_inputs.items()}

        l1_outputs = model(**inputs1, output_hidden_states=True)
        l2_outputs = model(**inputs2, output_hidden_states=True)
        v1 = l1_outputs['hidden_states'][-1][:,0]
        v2 = l2_outputs['hidden_states'][-1][:,0]

        cos_sim = cos_sim_fn(v1.unsqueeze(1), v2.unsqueeze(0)) / args.simcse_temp

        if parallel_inputs['input_ids'].shape[1] == 3:
          inputs3 = {k:v[:, 2] for k,v in parallel_inputs.items()}
          l3_outputs = model(**inputs3, output_hidden_states=True)
          v3 = l3_outputs['hidden_states'][-1][:,0]
          v1_v3_cos_sim = cos_sim_fn(v1.unsqueeze(1), v3.unsqueeze(0)) / args.simcse_temp
          cos_sim = torch.cat([cos_sim, v1_v3_cos_sim], 1)
          # add weights here if needed

        simcse_labels = torch.arange(cos_sim.size(0)).long().to(args.device)
        simcse_loss = simcse_loss_fn(cos_sim, simcse_labels)

        loss = args.alpha * loss + (1-args.alpha) * simcse_loss

      if args.extra_mlm:
        try:
          mlm_batch = next(mlm_iterator)
        except StopIteration:
          mlm_iterator = iter(mlm_dataloader)
          mlm_batch = next(mlm_iterator)
        mlm_inputs = {"input_ids": mlm_batch[0].to(args.device), "attention_mask": mlm_batch[1].to(args.device), "labels": mlm_batch[3].to(args.device)}

        inputs = {k:v[:, 0] for k,v in mlm_inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        
        mlm_loss = outputs['loss']
        loss = args.alpha * loss + (1-args.alpha) * mlm_loss

      if args.n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu parallel training
      if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

      if args.fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
          scaled_loss.backward()
      else:
        loss.backward()

      tr_loss += loss.item()
      if (step + 1) % args.gradient_accumulation_steps == 0:
        if args.fp16:
          torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
          torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        model.zero_grad()
        global_step += 1
        del loss

        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
          # Log metrics
          tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
          tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
          logging_loss = tr_loss

          # Only evaluate on single GPU otherwise metrics may not average well
          if (args.local_rank == -1 and args.evaluate_during_training):
            if args.mlm:
              tb_writer.add_scalar('mlm_loss', mlm_loss, global_step)
            if args.nonsup_simcse or args.sup_simcse:
              tb_writer.add_scalar('simcse_loss', simcse_loss, global_step)
            if args.eval_during_train_on_dev:
              if args.eval_during_train_use_pred_dataset:
                for lang, ds in zip(args.predict_languages, args.predict_datasets):
                  results = evaluate(args, model, tokenizer, split='dev', dataset=ds, language=lang, lang2id=lang2id)
                  for key, value in results.items():
                    tb_writer.add_scalar("eval_{}/{}".format(key, ds), value, global_step)
              else:
                results = evaluate(args, model, tokenizer, split='dev', dataset=args.train_dataset, language=args.train_language, lang2id=lang2id)
                for key, value in results.items():
                  tb_writer.add_scalar("eval_{}".format(key), value, global_step)
            else:
              results = evaluate(args, model, tokenizer, split=args.train_split, dataset=args.train_dataset, language=args.train_language, lang2id=lang2id)
              for key, value in results.items():
                tb_writer.add_scalar("eval_{}".format(key), value, global_step)

        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
          if args.eval_test_set:
            output_predict_file = os.path.join(args.output_dir, 'eval_test_results')
            total = total_correct = 0.0
            with open(output_predict_file, 'a') as writer:
              writer.write('\n======= Predict using the model from checkpoint-{}:\n'.format(global_step))
              for language, ds in zip(args.predict_languages, args.predict_datasets):
                result = evaluate(args, model, tokenizer, split=args.test_split, dataset=ds, language=language, lang2id=lang2id, prefix='checkpoint-'+str(global_step))
                writer.write('{}={}\n'.format(ds, result['acc']))
                total += result['num']
                total_correct += result['correct']
              writer.write('total={}\n'.format(total_correct / total))

          if args.save_only_best_checkpoint:
            if args.eval_during_train_use_pred_dataset:
              accs = []
              f1s = []
              for language, ds in zip(args.predict_languages, args.predict_datasets):
                result = evaluate(args, model, tokenizer, split='dev', dataset=ds, language=language, lang2id=lang2id, prefix=str(global_step))
                accs.append(result['acc'])
                f1s.append(result['macro f1'])
              acc = statistics.mean(accs)
              f1 = statistics.mean(f1s)
              logger.info(" Dev accuracy {} = {}".format(args.predict_datasets, acc))
              logger.info(" Dev macro f1 {} = {}".format(args.predict_datasets, f1))
            else:
              result = evaluate(args, model, tokenizer, split='dev', dataset=args.train_dataset, language=args.train_language, lang2id=lang2id, prefix=str(global_step))
              logger.info(" Dev accuracy {} = {}".format(args.train_language, result['acc']))
              logger.info(" Dev macro f1 {} = {}".format(args.train_language, result['macro f1']))
              f1 = result['macro f1']
            if f1 > best_score:
              logger.info(" result['macro f1']={} > best_score={}".format(f1, best_score))
              output_dir = os.path.join(args.output_dir, "checkpoint-best")
              best_checkpoint = output_dir
              best_score = f1
              # Save model checkpoint
              if not os.path.exists(output_dir):
                os.makedirs(output_dir)
              model_to_save = (
                model.module if hasattr(model, "module") else model
              )  # Take care of distributed/parallel training
              model_to_save.save_pretrained(output_dir)
              tokenizer.save_pretrained(output_dir)

              torch.save(args, os.path.join(output_dir, "training_args.bin"))
              logger.info("Saving model checkpoint to %s", output_dir)

              torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
              torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
              logger.info("Saving optimizer and scheduler states to %s", output_dir)
          else:
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if not os.path.exists(output_dir):
              os.makedirs(output_dir)
            model_to_save = (
              model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)

      if args.max_steps > 0 and global_step > args.max_steps:
        epoch_iterator.close()
        break
    if args.max_steps > 0 and global_step > args.max_steps:
      train_iterator.close()
      break

  if args.local_rank in [-1, 0]:
    tb_writer.close()

  return global_step, tr_loss / global_step, best_score, best_checkpoint


def evaluate(args, model, tokenizer, split='train', dataset='arc', language='en', lang2id=None, prefix="", output_file=None, output_only_prediction=True):
  """Evalute the model."""

  model.eval()

  eval_task_names = (args.task_name,)
  eval_outputs_dirs = (args.output_dir,)

  compute_preds = get_compute_preds(args, tokenizer, model, dataset)
  compute_loss = get_compute_loss(args, tokenizer, model, dataset)

  results = {}
  for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
    labels_list = []
    if type(dataset) == list:
      eval_datasets = []
      shift = 0
      for ds in dataset:
        eval_dataset = load_and_cache_examples(args, eval_task, ds, tokenizer, split=split, lang2id=lang2id, evaluate=True, shift=shift)
        processor = PROCESSORS[args.task_name][ds]()
        labels_list.extend(processor.get_labels())
        n_labels = len(processor.get_labels())
        shift += n_labels
        eval_datasets.append(eval_dataset)
      eval_dataset = ConcatDataset(eval_datasets)
    else:
      eval_dataset = load_and_cache_examples(args, eval_task, dataset, tokenizer, split=split, lang2id=lang2id, evaluate=True)
      processor = PROCESSORS[args.task_name][dataset]()
      labels_list.extend(processor.get_labels())

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1:
      model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} {} *****".format(prefix, dataset))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    sentences = None
    for batch in tqdm(eval_dataloader, desc="Evaluating", dynamic_ncols=True):
      model.eval()
      batch = tuple(t.to(args.device) for t in batch)

      with torch.no_grad():
        inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
        all_shifts = batch[4]
        all_ends = batch[5]
        if args.model_type != "distilbert":
          inputs["token_type_ids"] = (
            batch[2] if args.model_type in ["bert"] else None
          )  # XLM and DistilBERT don't use segment_ids
        if args.model_type == "xlm":
          inputs["langs"] = batch[6]
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs['hidden_states'][-1]

        tmp_eval_loss = compute_loss(model, logits, batch[3], all_shifts, all_ends)
        eval_loss += tmp_eval_loss.mean().item()

        l_mask = batch[3] == -100
      nb_eval_steps += 1
      if preds is None:
        preds = compute_preds(logits[~l_mask], all_shifts, all_ends).detach().cpu().numpy()
        out_label_ids = inputs["labels"][~l_mask].detach().cpu().numpy()
        if output_file:
          sentences = inputs["input_ids"].detach().cpu().numpy()
      else:
        preds = np.append(preds, compute_preds(logits[~l_mask], all_shifts, all_ends).detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(out_label_ids, inputs["labels"][~l_mask].detach().cpu().numpy(), axis=0)
        if output_file:
          sentences = np.append(sentences, inputs["input_ids"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    # if args.output_mode == "classification":
    #   preds = compute_preds(preds)
    #   # preds = np.argmax(preds, axis=1)
    # else:
    #   raise ValueError("No other `output_mode` for XNLI.")
    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    if output_file:
      logger.info("***** Save prediction ******")
      print(f'output file: {output_file}')
      with open(output_file, 'w') as fout:
        pad_token_id = tokenizer.pad_token_id
        sentences = sentences.astype(int).tolist()
        sentences = [[w for w in s if w != pad_token_id]for s in sentences]
        sentences = [tokenizer.convert_ids_to_tokens(s) for s in sentences]
        #fout.write('Prediction\tLabel\tSentences\n')
        for p, l, s in zip(list(preds), list(out_label_ids), sentences):
          p = labels_list[int(p)]
          l = labels_list[int(l)]
          s = ' '.join(s)
          if output_only_prediction:
            fout.write(str(p) + '\n')
          else:
            fout.write('{}\t{}\t{}\n'.format(p, l, s))

      cm = sklearn.metrics.confusion_matrix(out_label_ids, preds, list(range(len(labels_list))))
      with open(os.path.splitext(output_file)[0]+'_cm.txt', 'w') as f:
        f.write(' \t' + '\t'.join([f'pred_{i}' for i in labels_list]) + '\n')
        for i, label in enumerate(labels_list):
          f.write(f'true_{label}\t' + '\t'.join([str(v) for v in cm[i]]) + '\n')
    logger.info("***** Eval results {} {} *****".format(prefix, dataset))
    for key in sorted(result.keys()):
      logger.info("  %s = %s", key, str(result[key]))

  return results


def load_and_cache_examples(args, task, dataset, tokenizer, split='train', lang2id=None, evaluate=False, shift=0):
  # Make sure only the first process in distributed training process the
  # dataset, and the others will use the cache
  if args.local_rank not in [-1, 0] and not evaluate:
    torch.distributed.barrier()

  processor = PROCESSORS[task][dataset]()
  n_labels = len(processor.get_labels())
  language = processor.language
  output_mode = "classification"
  # Load data features from cache or dataset file
  lc = '_lc' if args.do_lower_case else ''
  if evaluate:
    mlm = 'no_mlm'
    da = ''
  else:
    if args.mlm:
      mlm = f'w_{args.mlm_probability}mlm'
    else:
      mlm = 'no_mlm'
    if args.robust == 'rs_da':
      da = f'_w_da_{args.robust_samples}'
    else:
      da = ''

  cache_model_name_or_path = list(filter(lambda x: x and 'checkpoint' not in x, args.model_name_or_path.split("/")))[-1]

  cached_features_file = os.path.join(
    args.data_dir,
    "cached_{}_{}_{}_{}_{}_{}_{}{}{}".format(
      dataset,
      split,
      cache_model_name_or_path,
      str(args.max_seq_length),
      str(task),
      str(language),
      mlm,
      lc,
      da,
    ),
  )
  if os.path.exists(cached_features_file) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)
  else:
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
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

    if da == f'_w_da_{args.robust_samples}':
      examples = perturb(examples, args.robust_samples, args.robust_neighbors)

    features = convert_examples_to_stance_features(
      examples,
      tokenizer,
      task,
      label_list=label_list,
      max_length=args.max_seq_length,
      output_mode=output_mode,
      pad_on_left=False,
      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
      pad_token_segment_id=0,
      lang2id=lang2id,
      mlm=args.mlm if not evaluate else False,
      mlm_probability=args.mlm_probability,
    )
    if args.local_rank in [-1, 0]:
      logger.info("Saving features into cached file %s", cached_features_file)
      torch.save(features, cached_features_file)

  # Make sure only the first process in distributed training process the
  # dataset, and the others will use the cache
  if args.local_rank == 0 and not evaluate:
    torch.distributed.barrier()

  # Convert to Tensors and build dataset
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
  all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
  if output_mode == "classification":
    all_labels = torch.tensor([[i if i == -100 else i + shift for i in f.label] for f in features], dtype=torch.long)
  else:
    raise ValueError("No other `output_mode` for {}.".format(args.task_name))
  if args.mlm and not evaluate:
    all_mlm_labels = torch.tensor([f.mlm_labels for f in features], dtype=torch.long)
  all_shifts = torch.tensor([shift for f in features], dtype=torch.long)
  all_ends = torch.tensor([shift+n_labels for f in features], dtype=torch.long)

  if args.mlm and not evaluate:
    if args.model_type == 'xlm':
      all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
      dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_shifts, all_ends, all_mlm_labels, all_langs)
    else:
      dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_shifts, all_ends, all_mlm_labels)
  else:
    if args.model_type == 'xlm':
      all_langs = torch.tensor([f.langs for f in features], dtype=torch.long)
      dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_shifts, all_ends, all_langs)
    else:
      dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels, all_shifts, all_ends)
  return dataset


def load_and_cache_parallel_examples(args, dataset, tokenizer, split='train', evaluate=False):
  # Make sure only the first process in distributed training process the
  # dataset, and the others will use the cache
  if args.local_rank not in [-1, 0] and not evaluate:
    torch.distributed.barrier()

  processor = PROCESSORS['parallel'][dataset]()
  language = processor.language
  # Load data features from cache or dataset file
  lc = '_lc' if args.do_lower_case else ''

  cache_model_name_or_path = list(filter(lambda x: x and 'checkpoint' not in x, args.model_name_or_path.split("/")))[-1]

  cached_features_file = os.path.join(
    args.data_dir,
    "cached_{}_{}_{}_{}_{}_{}_{}".format(
      dataset,
      split,
      cache_model_name_or_path,
      str(args.max_seq_length),
      'parallel',
      str(language),
      lc,
    ),
  )
  if os.path.exists(cached_features_file) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)
  else:
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
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

    features = convert_examples_to_parallel_features(
      examples,
      tokenizer,
      max_length=args.max_seq_length,
      pad_on_left=False,
      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
      pad_token_segment_id=0,
    )
    if args.local_rank in [-1, 0]:
      logger.info("Saving features into cached file %s", cached_features_file)
      torch.save(features, cached_features_file)

  # Make sure only the first process in distributed training process the
  # dataset, and the others will use the cache
  if args.local_rank == 0 and not evaluate:
    torch.distributed.barrier()

  # Convert to Tensors and build dataset
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
  all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
  if args.model_type == 'xlm':
    raise NotImplementedError('xlm model not supported')
  else:
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids)
  return dataset


def load_and_cache_mlm_examples(args, dataset, tokenizer, split='train', evaluate=False):
  # Make sure only the first process in distributed training process the
  # dataset, and the others will use the cache
  if args.local_rank not in [-1, 0] and not evaluate:
    torch.distributed.barrier()

  processor = PROCESSORS['mlm'][dataset]()
  language = processor.language
  # Load data features from cache or dataset file
  lc = '_lc' if args.do_lower_case else ''

  cache_model_name_or_path = list(filter(lambda x: x and 'checkpoint' not in x, args.model_name_or_path.split("/")))[-1]

  cached_features_file = os.path.join(
    args.data_dir,
    "cached_{}_{}_{}_{}_{}_{}_{}_{}".format(
      dataset,
      split,
      cache_model_name_or_path,
      str(args.max_seq_length),
      'extra_mlm',
      str(args.mlm_probability),
      str(language),
      lc,
    ),
  )
  if os.path.exists(cached_features_file) and not args.overwrite_cache:
    logger.info("Loading features from cached file %s", cached_features_file)
    features = torch.load(cached_features_file)
  else:
    logger.info("Creating features from dataset file at %s", args.data_dir)
    label_list = processor.get_labels()
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

    features = convert_examples_to_mlm_features(
      examples,
      tokenizer,
      max_length=args.max_seq_length,
      pad_on_left=False,
      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
      pad_token_segment_id=0,
    )
    if args.local_rank in [-1, 0]:
      logger.info("Saving features into cached file %s", cached_features_file)
      torch.save(features, cached_features_file)

  # Make sure only the first process in distributed training process the
  # dataset, and the others will use the cache
  if args.local_rank == 0 and not evaluate:
    torch.distributed.barrier()

  # Convert to Tensors and build dataset
  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
  all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
  all_mlm_labels = torch.tensor([f.label for f in features], dtype=torch.long)
  if args.model_type == 'xlm':
    raise NotImplementedError('xlm model not supported')
  else:
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_mlm_labels)
  return dataset



def main():
  parser = argparse.ArgumentParser()

  # Required parameters
  parser.add_argument(
    "--data_dir",
    default=None,
    type=str,
    required=True,
    help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
  )
  parser.add_argument(
    "--model_type",
    default=None,
    type=str,
    required=True,
    help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
  )
  parser.add_argument(
    "--model_name_or_path",
    default=None,
    type=str,
    required=True,
    help="Path to pre-trained model or shortcut name",
  )
  parser.add_argument(
    "--train_dataset",
    default=["arc"],
    nargs="*",
    type=str,
    help="Train dataset(s)."
  )
  parser.add_argument(
    "--predict_datasets",
    default=["arc"],
    nargs="*",
    type=str,
    help="prediction dataset(s)"
  )
  parser.add_argument(
    "--output_dir",
    default=None,
    type=str,
    required=True,
    help="The output directory where the model predictions and checkpoints will be written.",
  )
  parser.add_argument(
    "--task_name",
    default="stance",
    type=str,
    required=True,
    help="The task name",
  )

  # Other parameters
  parser.add_argument(
    "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
  )
  parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
  )
  parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
  )
  parser.add_argument(
    "--max_seq_length",
    default=128,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
  )
  parser.add_argument(
    "--loss_fn",
    choices=['bce', 'cross_entropy'],
    default="bce",
    help="Loss function to use"
  )
  parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
  parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
  parser.add_argument("--do_predict", action="store_true", help="Whether to run prediction.")
  parser.add_argument("--do_predict_dev", action="store_true", help="Whether to run prediction.")
  parser.add_argument("--init_checkpoint", type=str, default=None, help="initial checkpoint for predicting the dev set")
  parser.add_argument(
    "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
  )
  parser.add_argument("--eval_during_train_on_dev", action="store_true", help="Run eval on dev set during training")
  parser.add_argument(
    "--eval_during_train_use_pred_dataset", action="store_true", help="Use pred dataset for eval during training"
  )
  parser.add_argument('--mlm', action='store_true', help="use mlm in loss")
  parser.add_argument('--mlm_probability', default=0.105, type=float)
  parser.add_argument('--alpha', default=0.5, type=float, help="Balance between label loss and mlm")
  parser.add_argument('--negative_samples', default=0, type=int, help="Number of negative samples to draw")
  parser.add_argument('--ds_weights', default='equal', choices=['equal', 'inverse_scaled', 'uncorrected_scaled', 'random', 'scaled'], help='how to weight the different datasets')
  parser.add_argument('--ds_alpha', default=0.3, type=float, help="alpha for use in computing dataset weights")
  parser.add_argument('--synonyms_file', default='./synonyms.pkl', help="File containing synonyms")
  parser.add_argument('--robust', default='none', choices=['none', 'rs_rp', 'rs_da'], help="implement robust training")
  parser.add_argument('--robust_size', type=float, required=False, help="size of ball to search for robust training")
  parser.add_argument('--robust_samples', type=int, default=3, help="number of samples to draw for robust training")
  parser.add_argument('--robust_neighbors', default='./counterfitted_neighbors.json', help="file containing neighbors for robust perturbation")
  parser.add_argument('--nonsup_simcse', action='store_true', help="use nonsupervised simcse in loss")
  parser.add_argument('--sup_simcse', action='store_true', help="use supervised simcse in loss")
  parser.add_argument('--simcse_temp', default=0.05, type=float)
  parser.add_argument('--parallel_dataset', help="parallel dataset for supervised simcse")
  parser.add_argument('--extra_mlm', action='store_true', help="use supervised simcse in loss")
  parser.add_argument('--mlm_dataset', help="mlm dataset for additional mlm")
  parser.add_argument(
    "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
  )
  parser.add_argument("--train_split", type=str, default="train", help="split of training set")
  parser.add_argument("--test_split", type=str, default="test", help="split of training set")

  parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
  parser.add_argument(
    "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
  )
  parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
  )
  parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
  parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
  parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
  parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
  parser.add_argument(
    "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
  )
  parser.add_argument(
    "--max_steps",
    default=-1,
    type=int,
    help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
  )
  parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

  parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
  parser.add_argument("--log_file", default="train", type=str, help="log file")
  parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
  parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
  )
  parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
  parser.add_argument(
    "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
  )
  parser.add_argument(
    "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
  )
  parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

  parser.add_argument(
    "--fp16",
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
  )
  parser.add_argument(
    "--eval_test_set",
    action="store_true",
    help="Whether to evaluate test set durinng training",
  )
  parser.add_argument(
    "--fp16_opt_level",
    type=str,
    default="O1",
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
  )
  parser.add_argument(
    "--save_only_best_checkpoint", action="store_true", help="save only the best checkpoint"
  )
  parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
  parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
  parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
  args = parser.parse_args()

  if (
    os.path.exists(args.output_dir)
    and os.listdir(args.output_dir)
    and args.do_train
    and not args.overwrite_output_dir
  ):
    raise ValueError(
      "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
        args.output_dir
      )
    )

  if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
    os.makedirs(args.output_dir)

  logging.basicConfig(handlers=[logging.FileHandler(os.path.join(args.output_dir, args.log_file)), logging.StreamHandler()],
                      format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                      datefmt='%m/%d/%Y %H:%M:%S',
                      level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
  logging.info("Input args: %r" % args)

  # Setup distant debugging if needed
  if args.server_ip and args.server_port:
    import ptvsd
    print("Waiting for debugger attach")
    ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
    ptvsd.wait_for_attach()

  # Setup CUDA, GPU & distributed training
  if args.local_rank == -1 or args.no_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
  else:  # Initializes the distributed backend which sychronizes nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend="nccl")
    args.n_gpu = 1
  args.device = device

  # Setup logging
  logger.warning(
    "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    args.local_rank,
    device,
    args.n_gpu,
    bool(args.local_rank != -1),
    args.fp16,
  )

  # Set seed
  set_seed(args)

  # Prepare dataset
  if args.task_name not in PROCESSORS:
    raise ValueError("Task not found: %s" % (args.task_name))
  processor = list(PROCESSORS[args.task_name].values())[0]()
  args.output_mode = "classification"
  label_list = processor.get_labels()
  num_labels = len(label_list)

  if args.negative_samples > 0:
    args.synonyms = pickle.load(open(args.synonyms_file, 'rb'))

  if args.robust == 'none':
    args.robust = None

  if args.robust == 'rs_da':
    args.gradient_accumulation_steps *= args.robust_samples

  # Load pretrained model and tokenizer
  # Make sure only the first process in distributed training loads model & vocab
  if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()

  args.model_type = args.model_type.lower()
  config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
  config = config_class.from_pretrained(
    args.config_name if args.config_name else args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=args.task_name,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )
  logger.info("config = {}".format(config))

  tokenizer = tokenizer_class.from_pretrained(
    args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    do_lower_case=args.do_lower_case,
    cache_dir=args.cache_dir if args.cache_dir else None,
  )

  lang2id = config.lang2id if args.model_type == "xlm" else None
  logger.info("lang2id = {}".format(lang2id))

  # Make sure only the first process in distributed training loads model & vocab
  if args.local_rank == 0:
    torch.distributed.barrier()
  logger.info("Training/evaluation parameters %s", args)

  args.train_language = []
  for train_ds in args.train_dataset:
    lang = PROCESSORS[args.task_name][train_ds].language
    args.train_language.append(lang)

  args.predict_languages = []
  for pred_ds in args.predict_datasets:
    lang = PROCESSORS[args.task_name][pred_ds].language
    args.predict_languages.append(lang)

  # Training
  if args.do_train:
    if args.init_checkpoint:
      logger.info("loading from folder {}".format(args.init_checkpoint))
      model = model_class.from_pretrained(
        args.init_checkpoint,
        config=config,
        cache_dir=args.init_checkpoint,
        )
    else:
      logger.info("loading from existing model {}".format(args.model_name_or_path))
      model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
      )
    model.to(args.device)
    train_datasets = []
    shift = 0
    for lang, train_ds in zip(args.train_language, args.train_dataset):
      train_dataset = load_and_cache_examples(args, args.task_name, train_ds, tokenizer, split=args.train_split, lang2id=lang2id, evaluate=False, shift=shift)
      train_datasets.append(train_dataset)

      processor = PROCESSORS[args.task_name][train_ds]()
      n_labels = len(processor.get_labels())
      shift += n_labels
    if args.ds_weights != 'equal':
      ds_weights = compute_ds_weights(args, train_datasets)
      weighted_train_datasets = []
      for dataset, weight in zip(train_datasets, ds_weights):
        ds_tensors = dataset.tensors
        n_examples = ds_tensors[0].shape[0]
        weight_tensor = torch.tensor(weight).repeat(n_examples)
        new_train_dataset = TensorDataset(*ds_tensors, weight_tensor)
        weighted_train_datasets.append(new_train_dataset)
      train_datasets = weighted_train_datasets
    train_dataset = ConcatDataset(train_datasets)
    if args.parallel_dataset is not None:
      parallel_dataset = load_and_cache_parallel_examples(args, args.parallel_dataset, tokenizer)
    if args.mlm_dataset is not None:
      mlm_dataset = load_and_cache_mlm_examples(args, args.mlm_dataset, tokenizer)
    global_step, tr_loss, best_score, best_checkpoint = train(args, train_dataset, parallel_dataset, mlm_dataset, model, tokenizer, lang2id)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    logger.info(" best checkpoint = {}, best score = {}".format(best_checkpoint, best_score))

  # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
  if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
      os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
      model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Load a trained model and vocabulary that you have fine-tuned
    model = model_class.from_pretrained(args.output_dir)
    tokenizer = tokenizer_class.from_pretrained(args.output_dir)
    model.to(args.device)

  # Evaluation
  results = {}
  if args.init_checkpoint:
    best_checkpoint = args.init_checkpoint
  elif os.path.exists(os.path.join(args.output_dir, 'checkpoint-best')):
    best_checkpoint = os.path.join(args.output_dir, 'checkpoint-best')
  else:
    best_checkpoint = args.output_dir
  best_score = 0
  if args.do_eval and args.local_rank in [-1, 0]:
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
      checkpoints = list(
        os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
      )
      logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
      global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
      prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

      model = model_class.from_pretrained(checkpoint)
      model.to(args.device)
      result = evaluate(args, model, tokenizer, split='dev', dataset=args.train_dataset, language=args.train_language, lang2id=lang2id, prefix=prefix)
      if result['acc'] > best_score:
        best_checkpoint = checkpoint
        best_score = result['acc']
      result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
      results.update(result)

    output_eval_file = os.path.join(args.output_dir, 'eval_results')
    with open(output_eval_file, 'w') as writer:
      for key, value in results.items():
        writer.write('{} = {}\n'.format(key, value))
      writer.write("Best checkpoint is {}, best accuracy is {}".format(best_checkpoint, best_score))
      logger.info("Best checkpoint is {}, best accuracy is {}".format(best_checkpoint, best_score))

  # Prediction
  if args.do_predict and args.local_rank in [-1, 0]:
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path if args.model_name_or_path else best_checkpoint, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(best_checkpoint)
    model.to(args.device)
    output_predict_file = os.path.join(args.output_dir, args.test_split + '_results.txt')
    total = total_correct = 0.0
    with open(output_predict_file, 'a') as writer:
      writer.write('======= Predict using the model from {} for {}:\n'.format(best_checkpoint, args.test_split))
      for language, ds in zip(args.predict_languages, args.predict_datasets):
        output_file = os.path.join(args.output_dir, 'test-{}.tsv'.format(language))
        result = evaluate(args, model, tokenizer, split=args.test_split, dataset=ds, language=language, lang2id=lang2id, prefix='best_checkpoint', output_file=output_file)
        writer.write('{}={}\n'.format(language, result['acc']))
        logger.info('{}={}'.format(language, result['acc']))
        total += result['num']
        total_correct += result['correct']
      writer.write('total={}\n'.format(total_correct / total))

  if args.do_predict_dev:
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path if args.model_name_or_path else best_checkpoint, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(best_checkpoint)
    model.to(args.device)
    output_predict_file = os.path.join(args.output_dir, 'dev_results')
    total = total_correct = 0.0
    with open(output_predict_file, 'w') as writer:
      writer.write('======= Predict using the model from {}:\n'.format(args.init_checkpoint))
      for language, ds in zip(args.predict_languages, args.predict_datasets):
        output_file = os.path.join(args.output_dir, 'dev-{}.tsv'.format(language))
        result = evaluate(args, model, tokenizer, split='dev', dataset=ds, language=language, lang2id=lang2id, prefix='best_checkpoint', output_file=output_file, output_only_prediction=False)
        writer.write('{}={}\n'.format(language, result['acc']))
        total += result['num']
        total_correct += result['correct']
      writer.write('total={}\n'.format(total_correct / total))

  return result


if __name__ == "__main__":
  main()

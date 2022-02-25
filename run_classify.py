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
from processors.twitter2015 import Twitter2015Processor

from processors.utils import (
  convert_stance_examples_to_mlm_features,
)
from processors.argmin import ArgMinProcessor
from processors.arc import ARCProcessor
from processors.fnc1 import FNC1Processor
from processors.iac1 import IAC1Processor
from processors.ibmcs import IBMCSProcessor
from processors.nlpcc import NLPCCProcessor
from processors.perspectrum import PerspectrumProcessor
from processors.scd import SCDProcessor
from processors.semeval2016t6 import SemEval2016t6Processor
from processors.snopes import SnopesProcessor
from processors.twitter2015 import Twitter2015Processor
from processors.twitter2017 import Twitter2017Processor
from processors.vast import VASTProcessor

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
  'stance': {'arc': ARCProcessor,
             'argmin': ArgMinProcessor,
             'fnc1': FNC1Processor,
             'iac1': IAC1Processor,
             'ibmcs': IBMCSProcessor,
             'nlpcc': NLPCCProcessor,
             'perspectrum': PerspectrumProcessor,
             'scd': SCDProcessor,
             'semeval2016t6': SemEval2016t6Processor,
             'snopes': SnopesProcessor,
             'twitter2015': Twitter2015Processor,
             'twitter2017': Twitter2017Processor,
             'vast': VASTProcessor}
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
      output[i] = torch.max(score[shift:end], axis=0)[1]
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

  if args.loss_fn == 'cross_entropy':
    def compute_loss(model, preds, labels, shifts, ends):
      embeded_labels = []
      for label in toked_labels:
        embed = torch.mean(model.roberta.embeddings(label)[0], axis=0)
        embeded_labels.append(embed)
      LE = torch.stack(embeded_labels).detach()

      scores = (preds @ LE.T).permute(0, 2, 1)
      loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(scores, labels)
      return loss
  elif args.loss_fn == 'bce':
    bce_loss = torch.nn.BCEWithLogitsLoss()
    def compute_loss(model, preds, labels, shifts, ends):
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
        idx = labels == i
        sc = scores[:, i][idx]
        if sc.shape[0] >= 1:
          loss += bce_loss(sc, torch.ones_like(sc))

        idx_n = labels != i
        idx_n = torch.logical_and(idx_n, shifts <= i)
        idx_n = torch.logical_and(idx_n, ends > i)
        sc = scores[:, i][idx_n]
        if sc.shape[0] >= 1:
          loss += bce_loss(sc, torch.zeros_like(sc))

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


def set_seed(args):
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  if args.n_gpu > 0:
    torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, lang2id=None):
  """Train the model."""
  if args.local_rank in [-1, 0]:
    tb_writer = SummaryWriter(f'runs/{args.output_dir.split("/")[-1]}')

  datasets = args.train_dataset
  compute_loss = get_compute_loss(args, tokenizer, model, datasets)

  args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
  train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
  train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

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
  model.zero_grad()
  train_iterator = trange(
    epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
  )
  set_seed(args)  # Added here for reproductibility
  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
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
      loss = compute_loss(model, outputs['hidden_states'][-1], batch[3], all_shifts, all_ends)

      if args.mlm:
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

        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
          # Log metrics
          tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
          tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
          logging_loss = tr_loss

          # Only evaluate on single GPU otherwise metrics may not average well
          if (args.local_rank == -1 and args.evaluate_during_training):
            tb_writer.add_scalar('mlm_loss', mlm_loss, global_step)
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
    if type(dataset) == list:
      eval_datasets = []
      shift = 0
      for ds in dataset:
        eval_dataset = load_and_cache_examples(args, eval_task, ds, tokenizer, split=split, lang2id=lang2id, evaluate=True, shift=shift)
        processor = PROCESSORS[args.task_name][ds]()
        n_labels = len(processor.get_labels())
        shift += n_labels
        eval_datasets.append(eval_dataset)
      eval_dataset = ConcatDataset(eval_datasets)
    else:
      eval_dataset = load_and_cache_examples(args, eval_task, dataset, tokenizer, split=split, lang2id=lang2id, evaluate=True)

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
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
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
          if args.mlm:
            inputs["langs"] = batch[7]
          else:
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
      with open(output_file, 'w') as fout:
        pad_token_id = tokenizer.pad_token_id
        sentences = sentences.astype(int).tolist()
        sentences = [[w for w in s if w != pad_token_id]for s in sentences]
        sentences = [tokenizer.convert_ids_to_tokens(s) for s in sentences]
        #fout.write('Prediction\tLabel\tSentences\n')
        for p, l, s in zip(list(preds), list(out_label_ids), sentences):
          p = tokenizer.convert_ids_to_tokens(int(p))
          l = tokenizer.convert_ids_to_tokens(int(l))
          s = ' '.join(s)
          if output_only_prediction:
            fout.write(str(p) + '\n')
          else:
            fout.write('{}\t{}\t{}\n'.format(p, l, s))
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
  mlm = 'w_mlm' if args.mlm else 'no_mlm'
  cache_model_name_or_path = list(filter(lambda x: x and 'checkpoint' not in x, args.model_name_or_path.split("/")))[-1]

  cached_features_file = os.path.join(
    args.data_dir,
    "cached_{}_{}_{}_{}_{}_{}_{}{}".format(
      dataset,
      split,
      cache_model_name_or_path,
      str(args.max_seq_length),
      str(task),
      str(language),
      mlm,
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

    features = convert_stance_examples_to_mlm_features(
      examples,
      tokenizer,
      label_list=label_list,
      max_length=args.max_seq_length,
      output_mode=output_mode,
      pad_on_left=False,
      pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
      pad_token_segment_id=0,
      lang2id=lang2id,
      mlm=args.mlm,
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
  if args.mlm:
    all_mlm_labels = torch.tensor([f.mlm_labels for f in features], dtype=torch.long)
  all_shifts = torch.tensor([shift for f in features], dtype=torch.long)
  all_ends = torch.tensor([shift+n_labels for f in features], dtype=torch.long)

  if args.mlm:
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
  parser.add_argument('--mlm_probability', default=0.105)
  parser.add_argument('--alpha', default=0.5, type=float, help="Balance between label loss and mlm")
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
    train_dataset = ConcatDataset(train_datasets)
    global_step, tr_loss, best_score, best_checkpoint = train(args, train_dataset, model, tokenizer, lang2id)
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
    model = model_class.from_pretrained(args.init_checkpoint)
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

import copy
import csv
import json
import logging
import torch
import random
from transformers import BertTokenizer, XLMTokenizer, XLMRobertaTokenizer
import ud_head

logger = logging.getLogger(__name__)


class InputExample(object):
  """
  A single training/test example for simple sequence classification.
  Args:
    guid: Unique id for the example.
    text_a: string. The untokenized text of the first sequence. For single
    sequence tasks, only this sequence must be specified.
    text_b: (Optional) string. The untokenized text of the second sequence.
    Only must be specified for sequence pair tasks.
    label: (Optional) string. The label of the example. This should be
    specified for train and dev examples, but not for test examples.
  """

  def __init__(self, guid, text_a, text_b=None, label=None, language=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.language = language

  def __repr__(self):
    return str(self.to_json_string())

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class StanceExample(InputExample):
  """
  A single training/text example for stance detection
  Args:
    guid: Unique id for the example.
    topic: string. The untokenized text of the first sequence. For single
    sequence tasks, only this sequence must be specified.
    text: string. The untokenized text of the second sequence.
    Only must be specified for sequence pair tasks.
    label: (Optional) string. The label of the example. This should be
    specified for train and dev examples, but not for test examples.
  """

  def __init__(self, guid, topic, text, label=None, language=None, context=None):
    self.guid = guid
    self.topic = topic
    self.text = text
    self.context = context
    self.label = label
    self.language = language


class TripleSentExample(InputExample):
  """
  A single training/text example with three sentences
  Args:
    guid: Unique id for the example.
    sent0: string. The untokenized text of the first sequence.
    sent1: string. The untokenized text of the second sequence.
    sent2: (Optional) string. The untokenized text of the third sequence.
    label: (Optional) string. The label of the example. This should be
    specified for train and dev examples, but not for test examples.
  """

  def __init__(self, guid, sent0, sent1, sent2=None, label=None, language=None):
    self.guid = guid
    self.sent0 = sent0
    self.sent1 = sent1
    self.sent2 = sent2
    self.label = label
    self.language = language


class UDExample(InputExample):
  """
  A single training/text example with three sentences
  Args:
    guid: Unique id for the example.
    sent0: string. The untokenized text of the first sequence.
    sent1: string. The untokenized text of the second sequence.
    sent2: (Optional) string. The untokenized text of the third sequence.
    label: (Optional) string. The label of the example. This should be
    specified for train and dev examples, but not for test examples.
  """

  def __init__(self, guid, text, tokens, head, deprel):
    self.guid = guid
    self.text = text
    self.tokens = tokens
    self.head = head
    self.deprel = deprel


class InputFeatures(object):
  """
  A single set of features of data.
  Args:
    input_ids: Indices of input sequence tokens in the vocabulary.
    attention_mask: Mask to avoid performing attention on padding token indices.
      Mask values selected in ``[0, 1]``:
      Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
    token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    label: Label corresponding to the input
    topic: topic for this input
  """

  def __init__(self, input_ids, attention_mask=None, token_type_ids=None, langs=None, label=None, topic=None):
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.token_type_ids = token_type_ids
    self.label = label
    self.langs = langs
    self.topic = topic

  def __repr__(self):
    return str(self.to_json_string())

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class StanceFeatures(InputFeatures):
  """
  A single set of features of data.
  Args:
    input_ids: Indices of input sequence tokens in the vocabulary.
    attention_mask: Mask to avoid performing attention on padding token indices.
      Mask values selected in ``[0, 1]``:
      Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
    token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    label: Label corresponding to the input
    mlm_labels: Label for mlm task
    topic: topic for this input
  """
  def __init__(self, input_ids, attention_mask=None, token_type_ids=None, langs=None, label=None, mlm_labels=None, topic=None):
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.token_type_ids = token_type_ids
    self.label = label
    self.langs = langs
    self.mlm_labels = mlm_labels
    self.topic = topic


class UDFeatures(InputFeatures):
  """
  A single set of features of data.
  Args:
    ids: Indices of input sequence tokens in the vocabulary.
    attention_mask: Mask to avoid performing attention on padding token indices.
      Mask values selected in ``[0, 1]``:
      Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
    ud_arc: the UD head of this UD-token
    ud_rel: The UD relation to the head
    tok_lens: Number of LM-tokens per UD-token
  """
  def __init__(self, ids, attention_mask, ud_arc, ud_rel, tok_lens):
    self.ids = ids
    self.attention_mask = attention_mask
    self.ud_arc = ud_arc
    self.ud_rel = ud_rel
    self.tok_lens = tok_lens


def convert_examples_to_features(
  examples,
  tokenizer,
  max_length=512,
  label_list=None,
  output_mode=None,
  pad_on_left=False,
  pad_token=0,
  pad_token_segment_id=0,
  mask_padding_with_zero=True,
  lang2id=None,
):
  """
  Loads a data file into a list of ``InputFeatures``
  Args:
    examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
    tokenizer: Instance of a tokenizer that will tokenize the examples
    max_length: Maximum example length
    task: GLUE task
    label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
    output_mode: String indicating the output mode. Either ``regression`` or ``classification``
    pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
    pad_token: Padding token
    pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
    mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
      and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
      actual values)
  Returns:
    If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
    containing the task-specific features. If the input is a list of ``InputExamples``, will return
    a list of task-specific ``InputFeatures`` which can be fed to the model.
  """
  # is_tf_dataset = False
  # if is_tf_available() and isinstance(examples, tf.data.Dataset):
  #   is_tf_dataset = True

  label_map = {label: i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d" % (ex_index))
    # if is_tf_dataset:
    #   example = processor.get_example_from_tensor_dict(example)
    #   example = processor.tfds_map(example)

    if isinstance(tokenizer, XLMTokenizer):
      inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, lang=example.language)
    else:
      inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length)

    input_ids = inputs["input_ids"]

    try:
      token_type_ids = inputs["token_type_ids"]
    except KeyError:
      token_type_ids = [pad_token_segment_id] * len(input_ids)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
      input_ids = ([pad_token] * padding_length) + input_ids
      attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
      token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
      input_ids = input_ids + ([pad_token] * padding_length)
      attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
      token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    if lang2id is not None:
      lid = lang2id.get(example.language, lang2id["en"])
    else:
      lid = 0
    langs = [lid] * max_length

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
      len(attention_mask), max_length
    )
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
      len(token_type_ids), max_length
    )

    if output_mode == "classification":
      label = label_map[example.label]
    elif output_mode == "regression":
      label = float(example.label)
    else:
      raise KeyError(output_mode)

    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
      logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
      logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
      logger.info("label: %s (id = %d)" % (example.label, label))
      logger.info("language: %s, (lid = %d)" % (example.language, lid))

    features.append(
      InputFeatures(
        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, langs=langs, label=label
      )
    )
  return features


def mask_tokens(inputs, tokenizer, mlm_probability):
    """ Prepare masked tokens inputs/labels for masked language modeling, replace with only mask token """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.copy()
    n_tokens = len(inputs)
    max_masked = int(mlm_probability * n_tokens)
    if max_masked < 1:
      return inputs, [-100] * n_tokens

    special_tokens = set(map(tokenizer.convert_tokens_to_ids, tokenizer.special_tokens_map.values()))
    n_masked = random.randint(1, max_masked)
    masked_indices = [1]*n_masked + [0]*(n_tokens-n_masked)
    random.shuffle(masked_indices)
    for i, mask in enumerate(masked_indices):
      if mask == 0 or inputs[i] in special_tokens:
        labels[i] = -100  # We only compute loss on masked tokens
      else:
        inputs[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels


def order_sort(order, text, topic=None, context=None):
  if context is not None:
    text_idx = order.index('text')
    topic_idx = order.index('topic')
    context_idx = order.index('context')
    if text_idx < topic_idx:
      if context_idx < text_idx:
        return context, text, topic
      elif context_idx < topic_idx:
        return text, context, topic
      else:
        return text, topic, context
    else:
      if context_idx < topic_idx:
        return context, topic, text
      elif context_idx < text_idx:
        return topic, context, text
      else:
        return topic, text, context
  elif topic is not None:
    text_idx = order.index('text')
    topic_idx = order.index('topic')
    if text_idx < topic_idx:
      return text, topic
    else:
      return topic, text
  else:
    return text


def convert_examples_to_stance_features(
  examples,
  tokenizer,
  task,
  variant=0,
  context=False,
  max_length=512,
  label_list=None,
  output_mode=None,
  pad_on_left=False,
  pad_token=0,
  pad_token_segment_id=0,
  mask_padding_with_zero=True,
  lang2id=None,
  mlm=False,
  mlm_probability=0,
):
  """
  Loads a data file into a list of ``InputFeatures``
  Args:
    examples: List of ``StanceExamples`` or ``tf.data.Dataset`` containing the examples.
    tokenizer: Instance of a tokenizer that will tokenize the examples
    max_length: Maximum example length
    task: GLUE task
    variant: which variant of the task
    context: does the examples come with context
    label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
    output_mode: String indicating the output mode. Either ``regression`` or ``classification``
    pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
    pad_token: Padding token
    pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
    mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
      and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
      actual values)
  Returns:
    If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
    containing the task-specific features. If the input is a list of ``InputExamples``, will return
    a list of task-specific ``InputFeatures`` which can be fed to the model.
  """
  # is_tf_dataset = False
  # if is_tf_available() and isinstance(examples, tf.data.Dataset):
  #   is_tf_dataset = True

  if task == 'stance':
    if variant == 0:
      pattern0 = f'The stance of the following is {tokenizer.mask_token}'
      pattern1 = f'The stance of the following is {tokenizer.mask_token} the '
      pattern2 = f'The stance of the following is {tokenizer.mask_token} the where'
      patterns = [pattern0, pattern1, pattern2]
      patterns_len = list(map(len, patterns))
      text_index = len(tokenizer.encode('The stance of the following', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'The stance of the following is {tokenizer.mask_token} the', add_special_tokens=True)) - 1
      context_index = len(tokenizer.encode(f'The stance of the following is {tokenizer.mask_token} the where', add_special_tokens=True)) - 1
      order = ['text', 'topic', 'context']
    elif variant == 1:
      pattern0 = f'The opinion of the following text is {tokenizer.mask_token}'
      pattern1 = f'The opinion of the following text is {tokenizer.mask_token} the '
      pattern2 = f'The opinion of the following text is {tokenizer.mask_token} the where'
      patterns = [pattern0, pattern1, pattern2]
      patterns_len = list(map(len, patterns))
      text_index = len(tokenizer.encode('The opinion of the following text', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'The opinion of the following text is {tokenizer.mask_token} the', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'The opinion of the following text is {tokenizer.mask_token} the where', add_special_tokens=True)) - 1
      order = ['text', 'topic', 'context']
    elif variant == 2:
      pattern0 = f'The text is {tokenizer.mask_token}'
      pattern1 = f'The text is {tokenizer.mask_token} the '
      pattern2 = f'The text is {tokenizer.mask_token} the where'
      patterns = [pattern0, pattern1, pattern2]
      patterns_len = list(map(len, patterns))
      text_index = len(tokenizer.encode('The text ', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'The text is {tokenizer.mask_token} the', add_special_tokens=True)) - 1
      context_index = len(tokenizer.encode(f'The text is {tokenizer.mask_token} the where', add_special_tokens=True)) - 1
      order = ['text', 'topic', 'context']
    elif variant == 3:
      pattern0 = f'Stance of is {tokenizer.mask_token}'
      pattern1 = f'Stance of is {tokenizer.mask_token} the '
      pattern2 = f'Stance of is {tokenizer.mask_token} the where'
      patterns = [pattern0, pattern1, pattern2]
      patterns_len = list(map(len, patterns))
      text_index = len(tokenizer.encode('Stance of', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'Stance of is {tokenizer.mask_token} the', add_special_tokens=True)) - 1
      context_index = len(tokenizer.encode(f'Stance of is {tokenizer.mask_token} the where', add_special_tokens=True)) - 1
      order = ['text', 'topic', 'context']
    else:
      raise NotImplementedError('variant not implemented')
  else:
    raise NotImplementedError('task not implemented')

  label_map = {label: i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d" % (ex_index))
    # if is_tf_dataset:
    #   example = processor.get_example_from_tensor_dict(example)
    #   example = processor.tfds_map(example)

    if example.topic is None or example.topic == '':
      pattern = patterns[0]
      pattern_length = patterns_len[0]
      pattern_mode = 0
    elif not context or example.context is None or example.context == '':
      pattern = patterns[1]
      pattern_length = patterns_len[1]
      pattern_mode = 1
    else:
      pattern = patterns[2]
      pattern_length = patterns_len[2]
      pattern_mode = 2

    # truncate
    working_len = max_length - pattern_length
    if not (isinstance(tokenizer, XLMRobertaTokenizer) or isinstance(tokenizer, BertTokenizer)):
      raise NotImplementedError('This tokenizer is not supported')
    toked_text = tokenizer.encode_plus(example.text, add_special_tokens=False)
    text_len = len(toked_text['input_ids'])
    unmodified_toked_topic_ids = [pad_token]*(max_length)
    if pattern_mode > 0:
      toked_topic = tokenizer.encode_plus(example.topic, add_special_tokens=False)
      topic_len = len(toked_topic['input_ids'])
      unmodified_toked_topic_ids = toked_topic['input_ids'] + [pad_token]*(max_length-topic_len)
    if pattern_mode == 2:
      toked_context = tokenizer.encode_plus(example.context, add_special_tokens=False)
      context_len = len(toked_context['input_ids'])
    inputs = tokenizer.encode_plus(pattern, add_special_tokens=True, max_length=max_length)

    if pattern_mode == 0:
      if  text_len <= working_len:
        pass
      elif text_len > working_len:
        toked_text['input_ids'] = toked_text['input_ids'][:working_len]
        toked_text['attention_mask'] = toked_text['attention_mask'][:working_len]
      else:
        raise ValueError('This should not be reachable')

    elif pattern_mode == 1:
      if text_len + topic_len <= working_len:
        pass
      elif text_len > working_len/2 and topic_len > working_len/2:
        toked_text['input_ids'] = toked_text['input_ids'][:int(working_len/2)]
        toked_text['attention_mask'] = toked_text['attention_mask'][:int(working_len/2)]
        toked_topic['input_ids'] = toked_topic['input_ids'][:int(working_len/2)]
        toked_topic['attention_mask'] = toked_topic['attention_mask'][:int(working_len/2)]
      elif text_len > topic_len:
        toked_text['input_ids'] = toked_text['input_ids'][:working_len-topic_len]
        toked_text['attention_mask'] = toked_text['attention_mask'][:working_len-topic_len]
      elif topic_len > text_len:
        toked_topic['input_ids'] = toked_topic['input_ids'][:working_len-text_len]
        toked_topic['attention_mask'] = toked_topic['attention_mask'][:working_len-text_len]
      else:
        raise ValueError('This should not be reachable')

    elif pattern_mode == 2:
      if text_len + topic_len + context_len <= working_len:
        pass
      elif text_len > working_len/3 and topic_len > working_len/3 and context_len > working_len/3:
        toked_text['input_ids'] = toked_text['input_ids'][:int(working_len/3)]
        toked_text['attention_mask'] = toked_text['attention_mask'][:int(working_len/3)]
        toked_topic['input_ids'] = toked_topic['input_ids'][:int(working_len/3)]
        toked_topic['attention_mask'] = toked_topic['attention_mask'][:int(working_len/3)]
        toked_context['input_ids'] = toked_context['input_ids'][:int(working_len/3)]
        toked_context['attention_mask'] = toked_context['attention_mask'][:int(working_len/3)]
      elif text_len < working_len/3 and topic_len < working_len/3:
        leftover_len = working_len - text_len - topic_len
        toked_context['input_ids'] = toked_context['input_ids'][:leftover_len]
        toked_context['attention_mask'] = toked_context['attention_mask'][:leftover_len]
      elif text_len < working_len/3 and context_len < working_len/3:
        leftover_len = working_len - text_len - context_len
        toked_topic['input_ids'] = toked_topic['input_ids'][:leftover_len]
        toked_topic['attention_mask'] = toked_topic['attention_mask'][:leftover_len]
      elif topic_len < working_len/3 and context_len < working_len/3:
        leftover_len = working_len - topic_len - context_len
        toked_text['input_ids'] = toked_text['input_ids'][:leftover_len]
        toked_text['attention_mask'] = toked_text['attention_mask'][:leftover_len]
      elif text_len < working_len/3:
        leftover_len = working_len - text_len
        toked_topic['input_ids'] = toked_topic['input_ids'][:int(leftover_len/2)]
        toked_topic['attention_mask'] = toked_topic['attention_mask'][:int(leftover_len/2)]
        toked_context['input_ids'] = toked_context['input_ids'][:int(leftover_len/2)]
        toked_context['attention_mask'] = toked_context['attention_mask'][:int(leftover_len/2)]
      elif topic_len < working_len/3:
        leftover_len = working_len - topic_len
        toked_text['input_ids'] = toked_text['input_ids'][:int(leftover_len/2)]
        toked_text['attention_mask'] = toked_text['attention_mask'][:int(leftover_len/2)]
        toked_context['input_ids'] = toked_context['input_ids'][:int(leftover_len/2)]
        toked_context['attention_mask'] = toked_context['attention_mask'][:int(leftover_len/2)]
      elif context_len < working_len/3:
        leftover_len = working_len - context_len
        toked_text['input_ids'] = toked_text['input_ids'][:int(leftover_len/2)]
        toked_text['attention_mask'] = toked_text['attention_mask'][:int(leftover_len/2)]
        toked_topic['input_ids'] = toked_topic['input_ids'][:int(leftover_len/2)]
        toked_topic['attention_mask'] = toked_topic['attention_mask'][:int(leftover_len/2)]
      else:
        raise ValueError('This should not be reachable')

    if mlm:
      mlm_labels = [-100] * len(inputs['input_ids'])
      toked_text['input_ids'], text_label = mask_tokens(toked_text['input_ids'], tokenizer, mlm_probability)
      if pattern_mode > 0:
        toked_topic['input_ids'], topic_label = mask_tokens(toked_topic['input_ids'], tokenizer, mlm_probability)
      if pattern_mode == 2:
        toked_context['input_ids'], context_label = mask_tokens(toked_context['input_ids'], tokenizer, mlm_probability)

      if pattern_mode == 0:
        inputs['input_ids'] = inputs['input_ids'][:text_index] + toked_text['input_ids'] + inputs['input_ids'][text_index:]
        inputs['attention_mask'] = inputs['attention_mask'][:text_index] + toked_text['attention_mask'] + inputs['attention_mask'][text_index:]
        mlm_labels = mlm_labels[:text_index] + text_label + mlm_labels[text_index:]
      elif pattern_mode == 1:
        toked_first, toked_second = order_sort(order, toked_text, toked_topic)
        first_index, second_index = order_sort(order, text_index, topic_index)
        first_label, second_label = order_sort(order, text_label, topic_label)

        inputs['input_ids'] = inputs['input_ids'][:first_index] + toked_first['input_ids'] + inputs['input_ids'][first_index:second_index] + toked_second['input_ids'] + inputs['input_ids'][second_index:]
        inputs['attention_mask'] = inputs['attention_mask'][:first_index] + toked_first['attention_mask'] + inputs['attention_mask'][first_index:second_index] + toked_second['attention_mask'] + inputs['attention_mask'][second_index:]
        mlm_labels = mlm_labels[:first_index] + first_label + mlm_labels[first_index:second_index] + second_label + mlm_labels[second_index:]
      elif pattern_mode == 2:
        toked_first, toked_second, toked_third = order_sort(order, toked_text, toked_topic, toked_context)
        first_index, second_index, third_index = order_sort(order, text_index, topic_index, context_index)
        first_label, second_label, third_label = order_sort(order, text_label, topic_label, context_label)

        inputs['input_ids'] = inputs['input_ids'][:first_index] + toked_first['input_ids'] + inputs['input_ids'][first_index:second_index] + toked_second['input_ids'] + inputs['input_ids'][second_index:third_index] + toked_third['input_ids'] + inputs['input_ids'][third_index:]
        inputs['attention_mask'] = inputs['attention_mask'][:first_index] + toked_first['attention_mask'] + inputs['attention_mask'][first_index:second_index] + toked_second['attention_mask'] + inputs['attention_mask'][second_index:third_index] + toked_third['attention_mask'] + inputs['attention_mask'][third_index:]
        mlm_labels = mlm_labels[:first_index] + first_label + mlm_labels[first_index:second_index] + second_label + mlm_labels[second_index:third_index] + third_label + mlm_labels[third_index:]
    else:
      if pattern_mode == 0:
        inputs['input_ids'] = inputs['input_ids'][:text_index] + toked_text['input_ids'] + inputs['input_ids'][text_index:]
        inputs['attention_mask'] = inputs['attention_mask'][:text_index] + toked_text['attention_mask'] + inputs['attention_mask'][text_index:]
      elif pattern_mode == 1:
        toked_first, toked_second = order_sort(order, toked_text, toked_topic)
        first_index, second_index = order_sort(order, text_index, topic_index)

        inputs['input_ids'] = inputs['input_ids'][:first_index] + toked_first['input_ids'] + inputs['input_ids'][first_index:second_index] + toked_second['input_ids'] + inputs['input_ids'][second_index:]
        inputs['attention_mask'] = inputs['attention_mask'][:first_index] + toked_first['attention_mask'] + inputs['attention_mask'][first_index:second_index] + toked_second['attention_mask'] + inputs['attention_mask'][second_index:]
      elif pattern_mode == 2:
        toked_first, toked_second, toked_third = order_sort(order, toked_text, toked_topic, toked_context)
        first_index, second_index, third_index = order_sort(order, text_index, topic_index, context_index)

        inputs['input_ids'] = inputs['input_ids'][:first_index] + toked_first['input_ids'] + inputs['input_ids'][first_index:second_index] + toked_second['input_ids'] + inputs['input_ids'][second_index:third_index] + toked_third['input_ids'] + inputs['input_ids'][third_index:]
        inputs['attention_mask'] = inputs['attention_mask'][:first_index] + toked_first['attention_mask'] + inputs['attention_mask'][first_index:second_index] + toked_second['attention_mask'] + inputs['attention_mask'][second_index:third_index] + toked_third['attention_mask'] + inputs['attention_mask'][third_index:]

    input_ids = inputs["input_ids"]
    try:
      input_ids.index(tokenizer.mask_token_id)
    except ValueError:
      input_ids[-1] = tokenizer.mask_token_id
      input_ids[-2] = tokenizer.sep_token_id

    token_type_ids = [pad_token_segment_id] * len(input_ids)  # only 1 sentence is created

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
      input_ids = ([pad_token] * padding_length) + input_ids
      attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
      token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
      if mlm:
        mlm_labels = ([-100] * padding_length) + mlm_labels
    else:
      input_ids = input_ids + ([pad_token] * padding_length)
      attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
      token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
      if mlm:
        mlm_labels = mlm_labels + ([-100] * padding_length)

    if lang2id is not None:
      lid = lang2id.get(example.language, lang2id["en"])
    else:
      lid = 0
    langs = [lid] * max_length

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
      len(attention_mask), max_length
    )
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
      len(token_type_ids), max_length
    )
    if mlm:
      assert len(mlm_labels) == max_length, "Error with mlm labels length {} vs {}".format(
        len(mlm_labels), max_length
      )

    label = [-100] * len(input_ids)
    label[input_ids.index(tokenizer.mask_token_id)] = label_map[example.label]

    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
      logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
      logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
      logger.info("label: %s" % " ".join([str(x) for x in label]))
      logger.info("language: %s, (lid = %d)" % (example.language, lid))
      if mlm:
        logger.info("mlm labels; %s" % " ".join([str(x) for x in mlm_labels]))

    if mlm:
      features.append(
        StanceFeatures(
          input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, langs=langs, label=label, mlm_labels=mlm_labels, topic=unmodified_toked_topic_ids
        )
      )
    else:
      features.append(
        InputFeatures(
          input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, langs=langs, label=label, topic=unmodified_toked_topic_ids
        )
      )
  return features


def convert_examples_to_stance_features_orig(
  examples,
  tokenizer,
  task,
  variant=0,
  context=False,
  max_length=512,
  label_list=None,
  output_mode=None,
  pad_on_left=False,
  pad_token=0,
  pad_token_segment_id=0,
  mask_padding_with_zero=True,
  lang2id=None,
  mlm=False,
  mlm_probability=0,
):
  """
  Loads a data file into a list of ``InputFeatures``
  Args:
    examples: List of ``StanceExamples`` or ``tf.data.Dataset`` containing the examples.
    tokenizer: Instance of a tokenizer that will tokenize the examples
    max_length: Maximum example length
    task: GLUE task
    variant: which variant of the task
    context: does the examples come with context
    label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
    output_mode: String indicating the output mode. Either ``regression`` or ``classification``
    pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
    pad_token: Padding token
    pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
    mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
      and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
      actual values)
  Returns:
    If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
    containing the task-specific features. If the input is a list of ``InputExamples``, will return
    a list of task-specific ``InputFeatures`` which can be fed to the model.
  """
  # is_tf_dataset = False
  # if is_tf_available() and isinstance(examples, tf.data.Dataset):
  #   is_tf_dataset = True

  if task == 'stance':
    if variant == 0:
      pattern = f'The stance of the following is {tokenizer.mask_token} the '
      pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
      text_index = len(tokenizer.encode('The stance of the following', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'The stance of the following is {tokenizer.mask_token} the', add_special_tokens=True)) - 1
      one_sided = False
      text_first = True
    elif variant == 1:
      pattern = f'The opinion of the following text is {tokenizer.mask_token} the '
      pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
      text_index = len(tokenizer.encode('The opinion of the following text', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'The opinion of the following text is {tokenizer.mask_token} the', add_special_tokens=True)) - 1
      one_sided = False
      text_first = True
    elif variant == 2:
      pattern = f'The text is {tokenizer.mask_token} the '
      pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
      text_index = len(tokenizer.encode('The text ', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'The text is {tokenizer.mask_token} the', add_special_tokens=True)) - 1
      one_sided = False
      text_first = True
    elif variant == 3:
      if context:
        pattern = f'Stance of is {tokenizer.mask_token} the where'
        pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
        text_index = len(tokenizer.encode('Stance of', add_special_tokens=True)) - 1
        topic_index = len(tokenizer.encode(f'Stance of is {tokenizer.mask_token} the', add_special_tokens=True)) - 1
        context_index = len(tokenizer.encode(f'Stance of is {tokenizer.mask_token} the where', add_special_tokens=True)) - 1
        one_sided = False
        text_first = True
      else:
        pattern = f'Stance of is {tokenizer.mask_token} the '
        pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
        text_index = len(tokenizer.encode('Stance of', add_special_tokens=True)) - 1
        topic_index = len(tokenizer.encode(f'Stance of is {tokenizer.mask_token} the', add_special_tokens=True)) - 1
        one_sided = False
        text_first = True
    elif variant == 4:
      pattern = f'is {tokenizer.mask_token} the '
      pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
      text_index = len(tokenizer.encode('', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'is {tokenizer.mask_token} the', add_special_tokens=True)) - 1
      one_sided = False
      text_first = True
    elif variant == 5:
      pattern = f'Towards , is {tokenizer.mask_token}'
      pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
      text_index = len(tokenizer.encode(f'Towards ,', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'Towards ', add_special_tokens=True)) - 1
      one_sided = False
      text_first = False
    elif variant == 6:
      pattern = f'I guess that is {tokenizer.mask_token}'
      pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
      text_index = len(tokenizer.encode(f'I guess that ', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'I guess that is {tokenizer.mask_token}', add_special_tokens=True)) - 1
      one_sided = False
      text_first = True
    elif variant == 7:
      pattern = f'Predict that {tokenizer.mask_token}'
      pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
      text_index = len(tokenizer.encode(f'Predict that', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'Predict that {tokenizer.mask_token}', add_special_tokens=True)) - 1
      one_sided = False
      text_first = True
    elif variant == 8:
      pattern = f'Regarding , the text: is {tokenizer.mask_token}'
      pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
      text_index = len(tokenizer.encode(f'Regarding , the text:', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'Regarding', add_special_tokens=True)) - 1
      one_sided = False
      text_first = False
    elif variant == 9:
      pattern = f'We believe that is {tokenizer.mask_token}'
      pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
      text_index = len(tokenizer.encode(f'We believe that', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'We believe that is {tokenizer.mask_token}', add_special_tokens=True)) - 1
      one_sided = False
      text_first = True
    elif variant == 10:
      pattern = f'Topic: Text: Stance: {tokenizer.mask_token}'
      pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
      text_index = len(tokenizer.encode(f'Topic: Text:', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'Topic:', add_special_tokens=True)) - 1
      one_sided = False
      text_first = False
    elif variant == 11:
      pattern = f'Text: Topic: Stance: {tokenizer.mask_token}'
      pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
      text_index = len(tokenizer.encode(f'Text:', add_special_tokens=True)) - 1
      topic_index = len(tokenizer.encode(f'Text: Topic:', add_special_tokens=True)) - 1
      one_sided = False
      text_first = True
    else:
      raise IndexError('stance task does not have this variant')
  elif task == 'stance_qa':
    pattern = f'Question: What is the stance of with respect to ? Answer: {tokenizer.mask_token}'
    pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
    text_index = len(tokenizer.encode('Question: What is the stance of', add_special_tokens=True)) - 1
    topic_index = len(tokenizer.encode('Question: What is the stance of with respect to', add_special_tokens=True)) - 1
    one_sided = False
    text_first = True
  elif task == 'stance_reserved':
    pattern = f'<stance_tok_0><stance_tok_1><stance_tok_2><stance_tok_3><stance_tok_4>{tokenizer.mask_token}'
    pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
    text_index = len(tokenizer.encode('<stance_tok_0><stance_tok_1><stance_tok_2><stance_tok_3><stance_tok_4>', add_special_tokens=True)) - 1
    topic_index = len(tokenizer.encode(f'<stance_tok_0><stance_tok_1><stance_tok_2><stance_tok_3><stance_tok_4>{tokenizer.mask_token}', add_special_tokens=True)) - 1
    one_sided = False
    text_first = True
  elif task == 'zh_stance':
    pattern = f'以下的立场是{tokenizer.mask_token}'
    pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
    text_index = len(tokenizer.encode('以下', add_special_tokens=True)) - 1
    topic_index = len(tokenizer.encode(f'以下的立场是{tokenizer.mask_token}', add_special_tokens=True)) - 1
    one_sided = False
    text_first = True
  elif task == 'no_topic_stance':
    pattern = f'The stance of the following is {tokenizer.mask_token}'
    pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
    text_index = len(tokenizer.encode('The stance of the following', add_special_tokens=True)) - 1
    one_sided = True
  elif task == 'nli':
    pattern = f'This premise: {tokenizer.mask_token} this hypothesis: '
    pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
    text_index = len(tokenizer.encode('This premise: ', add_special_tokens=True)) - 1
    topic_index = len(tokenizer.encode(f'This premise: {tokenizer.mask_token} this hypothesis: ', add_special_tokens=True)) - 1
    one_sided = False
    text_first = True
  elif task == 'classification':
    pattern = f'This comment is {tokenizer.mask_token}'
    pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
    text_index = len(tokenizer.encode('This comment', add_special_tokens=True)) - 1
    one_sided = True
  elif task == 'pawx':
    pattern = f'Sentence 1 and sentence 2 {tokenizer.mask_token} paraphases of each other.'
    pattern_length = len(tokenizer.encode(pattern, add_special_tokens=True))
    text_index = len(tokenizer.encode('Sentence 1', add_special_tokens=True)) - 1
    topic_index = len(tokenizer.encode('Sentence 1 and sentence 2', add_special_tokens=True)) - 1
    one_sided = False
    text_first = True
  else:
    raise NotImplementedError(f'This task: {task} is not supported')

  label_map = {label: i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d" % (ex_index))
    # if is_tf_dataset:
    #   example = processor.get_example_from_tensor_dict(example)
    #   example = processor.tfds_map(example)

    # truncate
    working_len = max_length - pattern_length
    if not (isinstance(tokenizer, XLMRobertaTokenizer) or isinstance(tokenizer, BertTokenizer)):
      raise NotImplementedError('This tokenizer is not supported')
    toked_text = tokenizer.encode_plus(example.text, add_special_tokens=False)
    text_len = len(toked_text['input_ids'])
    toked_topic = tokenizer.encode_plus(example.topic, add_special_tokens=False)
    topic_len = len(toked_topic['input_ids'])
    unmodified_toked_topic_ids = toked_topic['input_ids'] + [pad_token]*(max_length-topic_len)
    if context and example.context is not None:
      toked_context = tokenizer.encode_plus(example.context, add_special_tokens=False)
      context_len = len(toked_context['input_ids'])
    inputs = tokenizer.encode_plus(pattern, add_special_tokens=True, max_length=max_length)
    if one_sided:
      if  text_len <= working_len:
        pass
      elif text_len > working_len:
        toked_text['input_ids'] = toked_text['input_ids'][:working_len]
        toked_text['attention_mask'] = toked_text['attention_mask'][:working_len]
      else:
        raise ValueError('This should not be reachable')

    elif context and example.context is not None:
      if text_len + topic_len + context_len <= working_len:
        pass
      elif text_len > working_len/3 and topic_len > working_len/3 and context_len > working_len/3:
        toked_text['input_ids'] = toked_text['input_ids'][:int(working_len/3)]
        toked_text['attention_mask'] = toked_text['attention_mask'][:int(working_len/3)]
        toked_topic['input_ids'] = toked_topic['input_ids'][:int(working_len/3)]
        toked_topic['attention_mask'] = toked_topic['attention_mask'][:int(working_len/3)]
        toked_context['input_ids'] = toked_context['input_ids'][:int(working_len/3)]
        toked_context['attention_mask'] = toked_context['attention_mask'][:int(working_len/3)]
      elif text_len < working_len/3 and topic_len < working_len/3:
        leftover_len = working_len - text_len - topic_len
        toked_context['input_ids'] = toked_context['input_ids'][:leftover_len]
        toked_context['attention_mask'] = toked_context['attention_mask'][:leftover_len]
      elif text_len < working_len/3 and context_len < working_len/3:
        leftover_len = working_len - text_len - context_len
        toked_topic['input_ids'] = toked_topic['input_ids'][:leftover_len]
        toked_topic['attention_mask'] = toked_topic['attention_mask'][:leftover_len]
      elif topic_len < working_len/3 and context_len < working_len/3:
        leftover_len = working_len - topic_len - context_len
        toked_text['input_ids'] = toked_text['input_ids'][:leftover_len]
        toked_text['attention_mask'] = toked_text['attention_mask'][:leftover_len]
      elif text_len < working_len/3:
        leftover_len = working_len - text_len
        toked_topic['input_ids'] = toked_topic['input_ids'][:int(leftover_len/2)]
        toked_topic['attention_mask'] = toked_topic['attention_mask'][:int(leftover_len/2)]
        toked_context['input_ids'] = toked_context['input_ids'][:int(leftover_len/2)]
        toked_context['attention_mask'] = toked_context['attention_mask'][:int(leftover_len/2)]
      elif topic_len < working_len/3:
        leftover_len = working_len - topic_len
        toked_text['input_ids'] = toked_text['input_ids'][:int(leftover_len/2)]
        toked_text['attention_mask'] = toked_text['attention_mask'][:int(leftover_len/2)]
        toked_context['input_ids'] = toked_context['input_ids'][:int(leftover_len/2)]
        toked_context['attention_mask'] = toked_context['attention_mask'][:int(leftover_len/2)]
      elif context_len < working_len/3:
        leftover_len = working_len - context_len
        toked_text['input_ids'] = toked_text['input_ids'][:int(leftover_len/2)]
        toked_text['attention_mask'] = toked_text['attention_mask'][:int(leftover_len/2)]
        toked_topic['input_ids'] = toked_topic['input_ids'][:int(leftover_len/2)]
        toked_topic['attention_mask'] = toked_topic['attention_mask'][:int(leftover_len/2)]
      else:
        raise ValueError('This should not be reachable')

    else:
      if text_len + topic_len <= working_len:
        pass
      elif text_len > working_len/2 and topic_len > working_len/2:
        toked_text['input_ids'] = toked_text['input_ids'][:int(working_len/2)]
        toked_text['attention_mask'] = toked_text['attention_mask'][:int(working_len/2)]
        toked_topic['input_ids'] = toked_topic['input_ids'][:int(working_len/2)]
        toked_topic['attention_mask'] = toked_topic['attention_mask'][:int(working_len/2)]
      elif text_len > topic_len:
        toked_text['input_ids'] = toked_text['input_ids'][:working_len-topic_len]
        toked_text['attention_mask'] = toked_text['attention_mask'][:working_len-topic_len]
      elif topic_len > text_len:
        toked_topic['input_ids'] = toked_topic['input_ids'][:working_len-text_len]
        toked_topic['attention_mask'] = toked_topic['attention_mask'][:working_len-text_len]
      else:
        raise ValueError('This should not be reachable')

    if one_sided:
      if mlm:
        mlm_labels = [-100] * len(inputs['input_ids'])
        toked_text['input_ids'], text_label = mask_tokens(toked_text['input_ids'], tokenizer, mlm_probability)

        inputs['input_ids'] = inputs['input_ids'][:text_index] + toked_text['input_ids'] + inputs['input_ids'][text_index:]
        inputs['attention_mask'] = inputs['attention_mask'][:text_index] + toked_text['attention_mask'] + inputs['attention_mask'][text_index:]
        mlm_labels = mlm_labels[:text_index] + text_label + mlm_labels[text_index:]
      else:
        inputs['input_ids'] = inputs['input_ids'][:text_index] + toked_text['input_ids'] + inputs['input_ids'][text_index:]
        inputs['attention_mask'] = inputs['attention_mask'][:text_index] + toked_text['attention_mask'] + inputs['attention_mask'][text_index:]
    elif context and example.context is not None:
      if mlm:
        mlm_labels = [-100] * len(inputs['input_ids'])
        toked_text['input_ids'], text_label = mask_tokens(toked_text['input_ids'], tokenizer, mlm_probability)
        toked_topic['input_ids'], topic_label = mask_tokens(toked_topic['input_ids'], tokenizer, mlm_probability)
        toked_context['input_ids'], context_label = mask_tokens(toked_context['input_ids'], tokenizer, mlm_probability)

        if text_first:
          toked_first = toked_text
          toked_second = toked_topic
          toked_third = toked_context
          first_index = text_index
          second_index = topic_index
          third_index = context_index
          first_label = text_label
          second_label = topic_label
          third_label = context_label
        else:
          toked_first = toked_topic
          toked_second = toked_text
          toked_third = toked_context
          first_index = topic_index
          second_index = text_index
          third_index = context_index
          first_label = topic_label
          second_label = text_label
          third_label = context_label

        inputs['input_ids'] = inputs['input_ids'][:first_index] + toked_first['input_ids'] + inputs['input_ids'][first_index:second_index] + toked_second['input_ids'] + inputs['input_ids'][second_index:third_index] + toked_third['input_ids'] + inputs['input_ids'][third_index:]
        inputs['attention_mask'] = inputs['attention_mask'][:first_index] + toked_first['attention_mask'] + inputs['attention_mask'][first_index:second_index] + toked_second['attention_mask'] + inputs['attention_mask'][second_index:third_index] + toked_third['attention_mask'] + inputs['attention_mask'][third_index:]
        mlm_labels = mlm_labels[:first_index] + first_label + mlm_labels[first_index:second_index] + second_label + mlm_labels[second_index:third_index] + third_label + mlm_labels[third_index:]
      else:
        if text_first:
          toked_first = toked_text
          toked_second = toked_topic
          toked_third = toked_context
          first_index = text_index
          second_index = topic_index
          third_index = context_index
        else:
          toked_first = toked_topic
          toked_second = toked_text
          toked_third = toked_context
          first_index = topic_index
          second_index = text_index
          third_index = context_index

        inputs['input_ids'] = inputs['input_ids'][:first_index] + toked_first['input_ids'] + inputs['input_ids'][first_index:second_index] + toked_second['input_ids'] + inputs['input_ids'][second_index:third_index] + toked_third['input_ids'] + inputs['input_ids'][third_index:]
        inputs['attention_mask'] = inputs['attention_mask'][:first_index] + toked_first['attention_mask'] + inputs['attention_mask'][first_index:second_index] + toked_second['attention_mask'] + inputs['attention_mask'][second_index:third_index] + toked_third['attention_mask'] + inputs['attention_mask'][third_index:]
    else:
      if mlm:
        mlm_labels = [-100] * len(inputs['input_ids'])
        toked_text['input_ids'], text_label = mask_tokens(toked_text['input_ids'], tokenizer, mlm_probability)
        toked_topic['input_ids'], topic_label = mask_tokens(toked_topic['input_ids'], tokenizer, mlm_probability)

        if text_first:
          toked_first = toked_text
          toked_second = toked_topic
          first_index = text_index
          second_index = topic_index
          first_label = text_label
          second_label = topic_label
        else:
          toked_first = toked_topic
          toked_second = toked_text
          first_index = topic_index
          second_index = text_index
          first_label = topic_label
          second_label = text_label

        inputs['input_ids'] = inputs['input_ids'][:first_index] + toked_first['input_ids'] + inputs['input_ids'][first_index:second_index] + toked_second['input_ids'] + inputs['input_ids'][second_index:]
        inputs['attention_mask'] = inputs['attention_mask'][:first_index] + toked_first['attention_mask'] + inputs['attention_mask'][first_index:second_index] + toked_second['attention_mask'] + inputs['attention_mask'][second_index:]
        mlm_labels = mlm_labels[:first_index] + first_label + mlm_labels[first_index:second_index] + second_label + mlm_labels[second_index:]
      else:
        if text_first:
          toked_first = toked_text
          toked_second = toked_topic
          first_index = text_index
          second_index = topic_index
        else:
          toked_first = toked_topic
          toked_second = toked_text
          first_index = topic_index
          second_index = text_index

        inputs['input_ids'] = inputs['input_ids'][:first_index] + toked_first['input_ids'] + inputs['input_ids'][first_index:second_index] + toked_second['input_ids'] + inputs['input_ids'][second_index:]
        inputs['attention_mask'] = inputs['attention_mask'][:first_index] + toked_first['attention_mask'] + inputs['attention_mask'][first_index:second_index] + toked_second['attention_mask'] + inputs['attention_mask'][second_index:]

    input_ids = inputs["input_ids"]
    try:
      input_ids.index(tokenizer.mask_token_id)
    except ValueError:
      input_ids[-1] = tokenizer.mask_token_id
      input_ids[-2] = tokenizer.sep_token_id

    token_type_ids = [pad_token_segment_id] * len(input_ids)  # only 1 sentence is created

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
      input_ids = ([pad_token] * padding_length) + input_ids
      attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
      token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
      if mlm:
        mlm_labels = ([-100] * padding_length) + mlm_labels
    else:
      input_ids = input_ids + ([pad_token] * padding_length)
      attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
      token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
      if mlm:
        mlm_labels = mlm_labels + ([-100] * padding_length)

    if lang2id is not None:
      lid = lang2id.get(example.language, lang2id["en"])
    else:
      lid = 0
    langs = [lid] * max_length

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
      len(attention_mask), max_length
    )
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
      len(token_type_ids), max_length
    )
    if mlm:
      assert len(mlm_labels) == max_length, "Error with mlm labels length {} vs {}".format(
        len(mlm_labels), max_length
      )

    label = [-100] * len(input_ids)
    label[input_ids.index(tokenizer.mask_token_id)] = label_map[example.label]

    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
      logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
      logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
      logger.info("label: %s" % " ".join([str(x) for x in label]))
      logger.info("language: %s, (lid = %d)" % (example.language, lid))
      if mlm:
        logger.info("mlm labels; %s" % " ".join([str(x) for x in mlm_labels]))

    if mlm:
      features.append(
        StanceFeatures(
          input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, langs=langs, label=label, mlm_labels=mlm_labels, topic=unmodified_toked_topic_ids
        )
      )
    else:
      features.append(
        InputFeatures(
          input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, langs=langs, label=label, topic=unmodified_toked_topic_ids
        )
      )
  return features

def process_one_sentence(sentence, tokenizer, max_length, pad_on_left, pad_token, pad_token_segment_id, mask_padding_with_zero):
  toked_sent = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_length, truncation=True)
  input_ids = toked_sent["input_ids"]
  try:
    token_type_ids = toked_sent["token_type_ids"]
  except KeyError:
    token_type_ids = [pad_token_segment_id] * len(input_ids)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

  # Zero-pad sent1 up to the sequence length.
  padding_length = max_length - len(input_ids)
  if pad_on_left:
    input_ids = ([pad_token] * padding_length) + input_ids
    attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
    token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
  else:
    input_ids = input_ids + ([pad_token] * padding_length)
    attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
    token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

  assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
  assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
    len(attention_mask), max_length
  )
  assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
    len(token_type_ids), max_length
  )

  return input_ids, attention_mask, token_type_ids


def convert_examples_to_parallel_features(
  examples,
  tokenizer,
  max_length=512,
  pad_on_left=False,
  pad_token=0,
  pad_token_segment_id=0,
  mask_padding_with_zero=True,
):
  """
  Loads a data file into a list of ``InputFeatures``
  Args:
    examples: List of ``StanceExamples`` or ``tf.data.Dataset`` containing the examples.
    tokenizer: Instance of a tokenizer that will tokenize the examples
    max_length: Maximum example length
    pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
    pad_token: Padding token
    pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
    mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
      and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
      actual values)
  Returns:
    If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
    containing the task-specific features. If the input is a list of ``InputExamples``, will return
    a list of task-specific ``InputFeatures`` which can be fed to the model.
  """

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d" % (ex_index))
    # if is_tf_dataset:
    #   example = processor.get_example_from_tensor_dict(example)
    #   example = processor.tfds_map(example)

    # truncate
    working_len = max_length
    if not (isinstance(tokenizer, XLMRobertaTokenizer) or isinstance(tokenizer, BertTokenizer)):
      raise NotImplementedError('This tokenizer is not supported')

    sent1_input_ids, sent1_attention_mask, sent1_token_type_ids = process_one_sentence(example.sent0, tokenizer, max_length, pad_on_left, pad_token, pad_token_segment_id, mask_padding_with_zero)
    sent2_input_ids, sent2_attention_mask, sent2_token_type_ids = process_one_sentence(example.sent1, tokenizer, max_length, pad_on_left, pad_token, pad_token_segment_id, mask_padding_with_zero)
    if example.sent2 is not None:
      sent3_input_ids, sent3_attention_mask, sent3_token_type_ids = process_one_sentence(example.sent2, tokenizer, max_length, pad_on_left, pad_token, pad_token_segment_id, mask_padding_with_zero)

    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("input_ids: %s" % " ".join([str(x) for x in sent1_input_ids]))
      logger.info("sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(sent1_input_ids)))
      logger.info("attention_mask: %s" % " ".join([str(x) for x in sent1_attention_mask]))
      logger.info("token_type_ids: %s" % " ".join([str(x) for x in sent1_token_type_ids]))

    if example.sent2 is not None:
      input_ids = (sent1_input_ids, sent2_input_ids, sent3_input_ids)
      attention_mask = (sent1_attention_mask, sent2_attention_mask, sent3_attention_mask)
      token_type_ids = (sent1_token_type_ids, sent2_token_type_ids, sent3_token_type_ids)

      flipped_input_ids = (sent2_input_ids, sent1_input_ids, sent3_input_ids)
      flipped_attention_mask = (sent2_attention_mask, sent1_attention_mask, sent3_attention_mask)
      flipped_token_type_ids = (sent2_token_type_ids, sent1_token_type_ids, sent3_token_type_ids)
    else:
      input_ids = (sent1_input_ids, sent2_input_ids)
      attention_mask = (sent1_attention_mask, sent2_attention_mask)
      token_type_ids = (sent1_token_type_ids, sent2_token_type_ids)

      flipped_input_ids = (sent2_input_ids, sent1_input_ids)
      flipped_attention_mask = (sent2_attention_mask, sent1_attention_mask)
      flipped_token_type_ids = (sent2_token_type_ids, sent1_token_type_ids)

    features.append(
      InputFeatures(
        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
      )
    )

    features.append(
      InputFeatures(
        input_ids=flipped_input_ids, attention_mask=flipped_attention_mask, token_type_ids=flipped_token_type_ids,
      )
    )
  return features

def convert_examples_to_mlm_features(
  examples,
  tokenizer,
  max_length=512,
  pad_on_left=False,
  pad_token=0,
  pad_token_segment_id=0,
  mask_padding_with_zero=True,
  mlm_probability=0,
  join_examples=False
):

  """
  Loads a data file into a list of ``InputFeatures``
  Args:
    examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
    tokenizer: Instance of a tokenizer that will tokenize the examples
    max_length: Maximum example length
    pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
    pad_token: Padding token
    pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
    mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
      and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
      actual values)
    mlm_probability: The rate of tokens to mask
    join_examples: If set to ``True``, will combine examples so that each output is of max length
  Returns:
    If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
    containing the task-specific features. If the input is a list of ``InputExamples``, will return
    a list of task-specific ``InputFeatures`` which can be fed to the model.
  """
  features = []
  if join_examples:
    toked_text = []
    long = False
    for (ex_index, example) in enumerate(examples):
      if ex_index % 10000 == 0:
        logger.info("Writing example %d" % (ex_index))
      # if is_tf_dataset:
      #   example = processor.get_example_from_tensor_dict(example)
      #   example = processor.tfds_map(example)
      if not (isinstance(tokenizer, XLMRobertaTokenizer) or isinstance(tokenizer, BertTokenizer)):
        raise NotImplementedError('This tokenizer is not supported')

      enc = tokenizer.encode(example.text_a, add_special_tokens=False)
      toked_text += enc

      if len(toked_text) <= max_length-2:
        continue
      long = True

      while long:
        # truncate
        input_ids = toked_text[:max_length-2]
        toked_text = toked_text[max_length-2:]
        if len(toked_text) < max_length-2:
          long = False

        input_ids = [tokenizer.bos_token_id] + input_ids +[tokenizer.eos_token_id]

        input_ids, mlm_labels = mask_tokens(input_ids, tokenizer, mlm_probability)

        token_type_ids = [pad_token_segment_id] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
          input_ids = ([pad_token] * padding_length) + input_ids
          attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
          token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
          mlm_labels = ([-100] * padding_length) + mlm_labels
        else:
          input_ids = input_ids + ([pad_token] * padding_length)
          attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
          token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
          mlm_labels = mlm_labels + ([-100] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
          len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
          len(token_type_ids), max_length
        )
        assert len(mlm_labels) == max_length, "Error with mlm labels length {} vs {}".format(
          len(mlm_labels), max_length
        )

        if len(features) < 5:
          logger.info("*** Example ***")
          logger.info("guid: %s" % (len(features)))
          logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
          logger.info("sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
          logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
          logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
          logger.info("mlm labels; %s" % " ".join([str(x) for x in mlm_labels]))

        features.append(
          InputFeatures(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=mlm_labels
          )
        )
    return features
  else:
    for (ex_index, example) in enumerate(examples):
      if ex_index % 10000 == 0:
        logger.info("Writing example %d" % (ex_index))
      # if is_tf_dataset:
      #   example = processor.get_example_from_tensor_dict(example)
      #   example = processor.tfds_map(example)

      # truncate
      if not (isinstance(tokenizer, XLMRobertaTokenizer) or isinstance(tokenizer, BertTokenizer)):
        raise NotImplementedError('This tokenizer is not supported')
      toked_text = tokenizer.encode_plus(example.text_a, add_special_tokens=True, max_length=max_length, truncation=True)

      toked_text['input_ids'], mlm_labels = mask_tokens(toked_text['input_ids'], tokenizer, mlm_probability)

      input_ids = toked_text["input_ids"]

      try:
        token_type_ids = toked_text["token_type_ids"]
      except KeyError:
        token_type_ids = [pad_token_segment_id] * len(input_ids)

      # The mask has 1 for real tokens and 0 for padding tokens. Only real
      # tokens are attended to.
      attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

      # Zero-pad up to the sequence length.
      padding_length = max_length - len(input_ids)
      if pad_on_left:
        input_ids = ([pad_token] * padding_length) + input_ids
        attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
        token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        mlm_labels = ([-100] * padding_length) + mlm_labels
      else:
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        mlm_labels = mlm_labels + ([-100] * padding_length)

      assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
      assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
        len(attention_mask), max_length
      )
      assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
        len(token_type_ids), max_length
      )
      assert len(mlm_labels) == max_length, "Error with mlm labels length {} vs {}".format(
        len(mlm_labels), max_length
      )

      if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
        logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
        logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
        logger.info("mlm labels; %s" % " ".join([str(x) for x in mlm_labels]))

      features.append(
        InputFeatures(
          input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=mlm_labels
        )
      )
    return features


def convert_examples_to_ud_features(
  examples,
  tokenizer,
  max_length=512,
  pad_on_left=False,
  mask_padding_with_zero=True,
):
  """
  Loads a data file into a list of ``InputFeatures``
  Args:
    examples: List of ``UDExamples`` or ``tf.data.Dataset`` containing the examples.
    tokenizer: Instance of a tokenizer that will tokenize the examples
    max_length: Maximum example length
    pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
    pad_token: Padding token
    pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
    mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
      and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
      actual values)
  Returns:
    If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
    containing the task-specific features. If the input is a list of ``UDExamples``, will return
    a list of task-specific ``InputFeatures`` which can be fed to the model.
  """
  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d" % (ex_index))

    try:
      tokens = list(map(tokenizer.tokenize, example.tokens))
      orig_ids = [[tokenizer.cls_token_id]] + list(map(tokenizer.convert_tokens_to_ids, tokens)) + [[tokenizer.sep_token_id]]
      ids = [[tokenizer.cls_token_id]] + list(map(tokenizer.convert_tokens_to_ids, tokens)) + [[tokenizer.sep_token_id]]
      ids = [i for toks in ids for i in toks]
      ud_arc = [0] + list(map(int, example.head)) + [0]
      ud_rel = ['punct'] + example.deprel + ['punct']
      ud_rel = list(map(lambda x: ud_head.UD_LABEL_DICT[x.split(':')[0]], ud_rel))
      tok_lens = [1] + list(map(len, tokens)) + [1]
      attention_mask = [1 if mask_padding_with_zero else 0] * len(ids)
    except ValueError:
      continue

    # truncate
    if sum(tok_lens) > max_length:
      continue

    assert sum(tok_lens) == len(ids)

    # padding
    padding_length = max_length - len(ud_arc)
    ids_pad_length = max_length - len(ids)
    if pad_on_left:
      ids = ([tokenizer.pad_token_id] * ids_pad_length) + ids
      ud_arc = ([-100] * padding_length) + ud_arc
      ud_rel = ([-100] * padding_length) + ud_rel
      tok_lens = ([0] * padding_length) + tok_lens
      attention_mask = ([0 if mask_padding_with_zero else 1] * ids_pad_length) + attention_mask
    else:
      ids = ids + ([tokenizer.pad_token_id] * ids_pad_length)
      ud_arc =  ud_arc + ([-100] * padding_length)
      ud_rel = ud_rel + ([-100] * padding_length)
      tok_lens = tok_lens + ([0] * padding_length)
      attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * ids_pad_length)

    assert len(ids) == max_length, "Error with ids length {} vs {}".format(len(ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
      len(attention_mask), max_length
    )
    assert len(ud_arc) == max_length, "Error with ud_arc length {} vs {}".format(
      len(ud_arc), max_length
    )
    assert len(ud_rel) == max_length, "Error with ud_rel length {} vs {}".format(
      len(ud_rel), max_length
    )
    assert len(tok_lens) == max_length, "Error with tok_lens labels length {} vs {}".format(
      len(tok_lens), max_length
    )

    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("ids: %s" % " ".join([str(x) for x in ids]))
      logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
      logger.info("sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(ids)))
      logger.info("ud_arc: %s" % " ".join([str(x) for x in ud_arc]))
      logger.info("ud_rel: %s" % " ".join([str(x) for x in ud_rel]))
      logger.info("tok_lens; %s" % " ".join([str(x) for x in tok_lens]))

    features.append(
      UDFeatures(
        ids=ids, attention_mask=attention_mask, ud_arc=ud_arc, ud_rel=ud_rel, tok_lens=tok_lens
      )
    )
  return features


def convert_examples_to_ld_features(
  examples,
  tokenizer,
  max_length=512,
  pad_on_left=False,
  pad_token=0,
  pad_token_segment_id=0,
  mask_padding_with_zero=True,
):
  """
  Loads a data file into a list of ``InputFeatures``
  Args:
    examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
    tokenizer: Instance of a tokenizer that will tokenize the examples
    max_length: Maximum example length
    pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
    pad_token: Padding token
    pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
    mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
      and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
      actual values)
  Returns:
    If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
    containing the task-specific features. If the input is a list of ``InputExamples``, will return
    a list of task-specific ``InputFeatures`` which can be fed to the model.
  """
  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d" % (ex_index))

    working_len = max_length
    if not (isinstance(tokenizer, XLMRobertaTokenizer) or isinstance(tokenizer, BertTokenizer)):
      raise NotImplementedError('This tokenizer is not supported')

    input_ids, attention_mask, token_type_ids = process_one_sentence(example.text_a, tokenizer, max_length, pad_on_left, pad_token, pad_token_segment_id, mask_padding_with_zero)
    label = example.label

    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
      logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
      logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
      logger.info("label: %s" % (label))

    features.append(
      InputFeatures(
        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
      )
    )

  return features
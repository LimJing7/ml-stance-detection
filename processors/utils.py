import copy
import csv
import json
import logging
import torch
import random
from transformers import XLMTokenizer, XLMRobertaTokenizer

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

  def __init__(self, guid, text_a, text_b, label=None, language=None):
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

  def __init__(self, guid, topic, text, label=None, language=None):
    self.guid = guid
    self.topic = topic
    self.text = text
    self.label = label
    self.language = language

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
  """

  def __init__(self, input_ids, attention_mask=None, token_type_ids=None, langs=None, label=None):
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.token_type_ids = token_type_ids
    self.label = label
    self.langs = langs

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
  """
  def __init__(self, input_ids, attention_mask=None, token_type_ids=None, langs=None, label=None, mlm_labels=None):
    self.input_ids = input_ids
    self.attention_mask = attention_mask
    self.token_type_ids = token_type_ids
    self.label = label
    self.langs = langs
    self.mlm_labels = mlm_labels


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
    n_masked = random.randint(1, max_masked)
    masked_indices = [1]*n_masked + [0]*(n_tokens-n_masked)
    random.shuffle(masked_indices)
    for i, mask in enumerate(masked_indices):
      if mask == 0:
        labels[i] = -100  # We only compute loss on masked tokens
      else:
        inputs[i] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    return inputs, labels


def convert_stance_examples_to_mlm_features(
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

    # truncate
    working_len = max_length - 11  # 11 = pattern length
    if not isinstance(tokenizer, XLMRobertaTokenizer):
      raise NotImplementedError('This tokenizer is not supported')
    toked_text = tokenizer.encode_plus(example.text, add_special_tokens=False)
    text_len = len(toked_text['input_ids'])
    toked_topic = tokenizer.encode_plus(example.topic, add_special_tokens=False)
    topic_len = len(toked_topic['input_ids'])
    inputs = tokenizer.encode_plus(f'The stance of the following is {tokenizer.mask_token} the ', add_special_tokens=True, max_length=max_length)
    if  text_len + topic_len <= working_len:
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
      toked_topic['input_ids'] = toked_topic['input_ids'][:working_len-topic_len]
      toked_topic['attention_mask'] = toked_topic['attention_mask'][:working_len-topic_len]
    else:
      raise ValueError('This should not be reachable')

    if mlm:
      mlm_labels = [-100] * len(inputs['input_ids'])
      toked_text['input_ids'], text_label = mask_tokens(toked_text['input_ids'], tokenizer, mlm_probability)
      toked_topic['input_ids'], topic_label = mask_tokens(toked_topic['input_ids'], tokenizer, mlm_probability)

      inputs['input_ids'] = inputs['input_ids'][:7] + toked_text['input_ids'] + inputs['input_ids'][7:10] + toked_topic['input_ids'] + inputs['input_ids'][10:]
      inputs['attention_mask'] = inputs['attention_mask'][:7] + toked_text['attention_mask'] + inputs['attention_mask'][7:10] + toked_topic['attention_mask'] + inputs['attention_mask'][10:]
      mlm_labels = mlm_labels[:7] + text_label + mlm_labels[7:10] + topic_label + mlm_labels[10:]
    else:
      inputs['input_ids'] = inputs['input_ids'][:7] + toked_text['input_ids'] + inputs['input_ids'][7:10] + toked_topic['input_ids'] + inputs['input_ids'][10:]
      inputs['attention_mask'] = inputs['attention_mask'][:7] + toked_text['attention_mask'] + inputs['attention_mask'][7:10] + toked_topic['attention_mask'] + inputs['attention_mask'][10:]

    input_ids = inputs["input_ids"]
    try:
      input_ids.index(tokenizer.mask_token_id)
    except ValueError:
      input_ids[-1] = tokenizer.mask_token_id
      input_ids[-2] = tokenizer.sep_token_id

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
          input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, langs=langs, label=label, mlm_labels=mlm_labels
        )
      )
    else:
      features.append(
        InputFeatures(
          input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, langs=langs, label=label
        )
      )
  return features

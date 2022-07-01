# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" ASAP utils (dataset loading and evaluation) """


import pandas as pd
import logging
import os

from transformers import DataProcessor
from .utils import StanceExample

logger = logging.getLogger(__name__)


class ASAPProcessor(DataProcessor):
    """Processor for the ASAP dataset.
    data from https://github.com/Meituan-Dianping/asap
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    label_map = {-2: 'not mentioned',
                 -1: 'negative',
                 0: 'neutral',
                 1: 'positive'}
    language = 'zh'

    def __init__(self):
        pass

    def get_examples(self, data_dir, split='train'):
      """See base class."""
      examples = []
      lines = pd.read_csv(os.path.join(data_dir, f"asap-{split}.csv"), header=0)

      topics_map = {
        'Location#Transportation': 'transportation',
        'Location#Downtown': 'downtown',
        'Location#Easy_to_find': 'easy to find',
        'Service#Queue': 'queue',
        'Service#Hospitality': 'hospitality',
        'Service#Parking': 'parking',
        'Service#Timely': 'timely',
        'Price#Level': 'price level',
        'Price#Cost_effective': 'cost effectiveness',
        'Price#Discount': 'discount',
        'Ambience#Decoration': 'decoration',
        'Ambience#Noise': 'noise',
        'Ambience#Space': 'space',
        'Ambience#Sanitary': 'sanitary',
        'Food#Portion': 'food portion',
        'Food#Taste': 'food taste',
        'Food#Appearance': 'food appearance',
        'Food#Recommend': 'food recommend'
      }

      for (i, line) in enumerate(lines.iterrows()):
        for j, (header, topic) in enumerate(topics_map.items()):
            guid = "%s-%s" % (split, i*18+j)
            text = line[1]['review']
            label = ASAPProcessor.label_map[line[1][header]]
            assert isinstance(topic, str) and isinstance(text, str) and isinstance(label, str)
            examples.append(StanceExample(guid=guid, topic=topic, text=text, label=label))
      return examples

    def get_train_examples(self, data_dir):
        return self.get_examples(data_dir, split='train')

    def get_dev_examples(self, data_dir):
        return self.get_examples(data_dir, split='dev')

    def get_test_examples(self, data_dir, ):
        return self.get_examples(data_dir, split='test')

    def get_labels(self):
        """See base class."""
        return ["not mentioned", "negative", "neutral", "positive"]


asap_processors = {
    "stance": ASAPProcessor,
}

asap_output_modes = {
    "stance": "classification",
}

asap_tasks_num_labels = {
    "stance": 4,
}

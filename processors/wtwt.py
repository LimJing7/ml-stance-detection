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
""" r-ita utils (dataset loading and evaluation) """


import json
import logging
import os

from transformers import DataProcessor
from .utils import StanceExample

logger = logging.getLogger(__name__)


class WtwtProcessor(DataProcessor):
    """Processor for the wtwt dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    label_map = {'comment': 'commenting on',
                 'refute': 'refuting',
                 'support': 'supporting',
                 'unrelated': 'unrelated to'}
    premise_maps = {"AET_HUM": "Aetna acquiring Humana",
                    "ANTM_CI": "Anthem acquiring Cigna",
                    "CVS_AET": "CVS Health acquiring Aetna",
                    "CI_ESRX": "Cigna acquiring Express Scripts",
                    "FOXA_DIS": "Disney acquiring 21st Century Fox"}
    language = 'en'

    def __init__(self):
        pass

    def get_examples(self, data_dir, split='train'):
      """See base class."""
      examples = []
      data = json.load(open(os.path.join(data_dir, f"wtwt-{split}.json")))

      for (i, line) in enumerate(data):
        guid = "%s-%s" % (split, i)
        topic = WtwtProcessor.premise_maps[line['merger']]
        text = line['text']
        label = WtwtProcessor.label_map[line['stance']]

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
        return ["commenting on", "refuting", "supporting", "unrelated to"]  # changing labels require overwriting cache


wtwt_processors = {
    "stance": WtwtProcessor,
}

wtwt_output_modes = {
    "stance": "classification",
}

wtwt_tasks_num_labels = {
    "stance": 4,
}

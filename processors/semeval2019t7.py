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
""" SemEval 2019 task 7 utils (dataset loading and evaluation) """


import jsonlines
import logging
import os

from transformers import DataProcessor
from .utils import StanceExample

logger = logging.getLogger(__name__)


class SemEval2019t7Processor(DataProcessor):
    """Processor for the SemEval 2019 task 7 dataset.
    map from label generated by mdl-stance-robustness
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    label_map = {1: 'AGAINST',
                 2: 'DISCUSS',
                 0: 'FAVOR',
                 3: 'UNRELATED'}
    language = 'en'

    def __init__(self):
        pass

    def get_examples(self, data_dir, split='train'):
      """See base class."""
      examples = []
      lines = jsonlines.jsonlines.open(os.path.join(data_dir, f"semeval2019t7-{split}.json"))

      for (i, line) in enumerate(lines):
        guid = "%s-%s" % (split, i)
        topic = line['premise']
        # text = line['hypothesis']
        label = SemEval2019t7Processor.label_map[line['label']]
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
        return ["AGAINST", "DISCUSS", "FAVOR", "UNRELATED"]


semeval2019t7_processors = {
    "stance": SemEval2019t7Processor,
}

semeval2019t7_output_modes = {
    "stance": "classification",
}

semeval2019t7_tasks_num_labels = {
    "stance": 4,
}
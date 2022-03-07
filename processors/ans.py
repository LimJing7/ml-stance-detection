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
""" ANS utils (dataset loading and evaluation) """


import csv
import logging
import os

from transformers import DataProcessor
from .utils import StanceExample

logger = logging.getLogger(__name__)


class ANSProcessor(DataProcessor):
    """Processor for the ANS dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    label_map = {'disagree': 'contradiction of',
                 'other': 'other',
                 'agree': 'paraphrase of',
                 0: 'UNRELATED'}
    language = 'ar'

    def __init__(self):
        pass

    def get_examples(self, data_dir, split='train'):
      """See base class."""
      examples = []
      with open(os.path.join(data_dir, f"ans-{split}.csv")) as f:
        lines = csv.reader(f, delimiter=',', quotechar='"')
        header = next(lines)
        text_id = header.index('s1')
        topic_id = header.index('s2')
        label_id = header.index('stance')
        n_terms = len(header)

        for (i, line) in enumerate(lines):
            if len(line) != n_terms:
                print(len(line))
                raise NotImplementedError
            guid = "%s-%s" % (split, i)
            topic = line[topic_id]
            text = line[text_id]
            label = ANSProcessor.label_map[line[label_id]]
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
        return ["contradiction of", "other", "paraphrase of"]


ans_processors = {
    "stance": ANSProcessor,
}

ans_output_modes = {
    "stance": "classification",
}

ans_tasks_num_labels = {
    "stance": 4,
}

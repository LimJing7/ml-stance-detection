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
""" IndoLI utils (dataset loading and evaluation) """


import csv
import logging
import os

from transformers import DataProcessor
from .utils import StanceExample

from datasets import load_dataset

logger = logging.getLogger(__name__)


class IndonliProcessor(DataProcessor):
    """Processor for the indoli dataset.
    map from label generated by indonli
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    label_map = {0: 'entail',
                 2: 'contradict',
                 1: 'neutral',}
    language = 'id'

    def __init__(self):
        pass

    def get_examples(self, data_dir, split='train'):
        """See base class."""
        examples = []
        dataset = load_dataset('indonli', split=split)

        for i, example in enumerate(dataset):
            guid = "%s-%s" % (split, i)
            text = example['premise']
            topic = example['hypothesis']
            label = example['label']

            label = IndonliProcessor.label_map[label]
            assert isinstance(topic, str) and isinstance(text, str) and isinstance(label, str)
            examples.append(StanceExample(guid=guid, topic=topic, text=text, label=label))

        return examples

    def get_train_examples(self, data_dir):
        return self.get_examples(data_dir, split='train')

    def get_dev_examples(self, data_dir):
        return self.get_examples(data_dir, split='validation')

    def get_test_examples(self, data_dir, ):
        return self.get_examples(data_dir, split='test_expert')

    def get_labels(self):
        """See base class."""
        return ["entail", "contradict", "neutral"]


indonli_processors = {
    "stance": IndonliProcessor,
}

indonli_output_modes = {
    "stance": "classification",
}

indonli_tasks_num_labels = {
    "stance": 3,
}

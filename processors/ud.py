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
from .utils import UDExample

from datasets import load_dataset

logger = logging.getLogger(__name__)


class UDProcessor(DataProcessor):
    """Processor for the universal dependencies dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    language = 'id'

    def __init__(self):
        pass

    def get_examples(self, data_dir, ud_ds, split='train'):
        """See base class."""
        examples = []
        dataset = load_dataset('universal_dependencies', ud_ds, split=split)

        for i, example in enumerate(dataset):
            guid = "%s-%s" % (split, i)
            text = example['text']
            tokens = example['tokens']
            head = example['head']
            deprel = example['deprel']

            assert isinstance(text, str)
            try:
                list(map(int, head))
            except ValueError:
                pass

            examples.append(UDExample(guid=guid, text=text, tokens=tokens, head=head, deprel=deprel))

        return examples

    def get_train_examples(self, data_dir, ud_ds):
        return self.get_examples(data_dir, ud_ds, split='train')

    def get_dev_examples(self, data_dir, ud_ds):
        return self.get_examples(data_dir, ud_ds, split='validation')

    def get_test_examples(self, data_dir, ud_ds):
        return self.get_examples(data_dir, ud_ds, split='test')

    def get_labels(self):
        """See base class."""
        return ["entail", "contradict", "neutral"]


ud_processors = {
    "stance": UDProcessor,
}

ud_output_modes = {
    "stance": "classification",
}

ud_tasks_num_labels = {
    "stance": 3,
}

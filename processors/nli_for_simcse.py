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
""" NLIforSIMCSE utils (dataset loading and evaluation) """


import csv
import logging
import os

from transformers import DataProcessor
from .utils import TripleSentExample

from datasets import load_dataset

logger = logging.getLogger(__name__)


class NLIforSIMCSEProcessor(DataProcessor):
    """Processor for the NLI for SIMCSE dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    language = 'en'

    def __init__(self):
        pass

    def get_examples(self, data_dir, split='train'):
        """See base class."""
        examples = []
        with open(os.path.join(data_dir, "nli_for_simcse.csv")) as f:
            reader = csv.reader(f, delimiter=',', quotechar='"')

            header = next(reader)
            s0_id = header.index('sent0')
            s1_id = header.index('sent1')
            hn_id = header.index('hard_neg')

            for i, row in enumerate(reader):
                guid = "%s-%s" % (split, i)
                sent0 = row[s0_id]
                sent1 = row[s1_id]
                hard_neg = row[hn_id]

                examples.append(TripleSentExample(guid=guid, sent0=sent0, sent1=sent1, sent2=hard_neg))

        return examples

    def get_train_examples(self, data_dir):
        return self.get_examples(data_dir, split='train')

    def get_dev_examples(self, data_dir):
        raise NotImplementedError('no dev set')

    def get_test_examples(self, data_dir, ):
        raise NotImplementedError('no test set')

    def get_labels(self):
        """See base class."""
        return []


nli_for_simcse_processors = {
    "stance": NLIforSIMCSEProcessor,
}

nli_for_simcse_output_modes = {
    "stance": "classification",
}

nli_for_simcse_tasks_num_labels = {
    "stance": 0,
}

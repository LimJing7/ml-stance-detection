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
from .utils import TripleSentExample

from datasets import load_dataset

logger = logging.getLogger(__name__)


class ParallelNLIProcessor(DataProcessor):
    """Processor for the 15 way NLI dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    language = 'en-zh'

    def __init__(self):
        pass

    def get_examples(self, data_dir, split='train'):
        """See base class."""
        examples = []
        with open(os.path.join(data_dir, "xnli.15way.orig.tsv")) as f:
            reader = csv.reader(f, delimiter='\t', quotechar='"')

            header = next(reader)
            en_id = header.index('en')
            zh_id = header.index('zh')

            for i, row in enumerate(reader):
                guid = "%s-%s" % (split, i)
                en_text = row[en_id]
                zh_text = row[zh_id]

                examples.append(TripleSentExample(guid=guid, sent0=en_text, sent1=zh_text))

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


parallelnli_processors = {
    "stance": ParallelNLIProcessor,
}

parallelnli_output_modes = {
    "stance": "classification",
}

parallelnli_tasks_num_labels = {
    "stance": 0,
}

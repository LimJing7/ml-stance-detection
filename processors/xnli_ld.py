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
from .utils import InputExample


logger = logging.getLogger(__name__)


class XNLILDProcessor(DataProcessor):
    """Processor for the 15 way NLI dataset for training lang discriminator.
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
                en_guid = "%s-%s" % (split, i*2)
                zh_guid = "%s-%s" % (split, i*2+1)
                en_text = row[en_id]
                zh_text = row[zh_id]

                examples.append(InputExample(guid=en_guid, text_a=en_text, label=1))
                examples.append(InputExample(guid=zh_guid, text_a=zh_text, label=0))

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
    "stance": XNLILDProcessor,
}

parallelnli_output_modes = {
    "stance": "classification",
}

parallelnli_tasks_num_labels = {
    "stance": 2,
}

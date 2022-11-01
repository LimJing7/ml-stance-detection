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
""" NusaX utils (dataset loading and evaluation) """


import pandas as pd
import logging
import os

from transformers import DataProcessor
from .utils import StanceExample

logger = logging.getLogger(__name__)


class NusaXProcessor(DataProcessor):
    """Processor for the NusaX dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    label_map = {'negative': 'negative',
                'positive': 'positive',
                'neutral': 'neutral'}
    language = 'xx'

    def __init__(self, language):
        self.language = language

    def get_examples(self, data_dir, split='train'):
        """See base class."""
        examples = []
        lines = pd.read_csv(os.path.join(data_dir, f"nusax_{self.language}-{split}.csv"), quotechar='"')

        for (i, line) in enumerate(lines.iterrows()):
            guid = "%s-%s" % (split, i)
            text = line[1]['text']
            label = NusaXProcessor.label_map[str(line[1]['label'].strip())]
            assert isinstance(text, str) and isinstance(label, str)
            examples.append(StanceExample(guid=guid, topic='', text=text, label=label))
        return examples

    def get_train_examples(self, data_dir):
        return self.get_examples(data_dir, split='train')

    def get_dev_examples(self, data_dir):
        return self.get_examples(data_dir, split='dev')

    def get_test_examples(self, data_dir, ):
        return self.get_examples(data_dir, split='test')

    def get_labels(self):
        """See base class."""
        return ["negative", "positive", "neutral"]  # changing labels require overwriting cache


nusax_processors = {
    "stance": NusaXProcessor,
}

nusax_output_modes = {
    "stance": "classification",
}

nusax_tasks_num_labels = {
    "stance": 3,
}

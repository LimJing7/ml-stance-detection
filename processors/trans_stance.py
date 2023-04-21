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
""" NLPCC utils (dataset loading and evaluation) """


import logging
import pandas as pd
import os

from transformers import DataProcessor
from .utils import StanceExample

logger = logging.getLogger(__name__)


class TransStanceProcessor(DataProcessor):
    """Processor for the NLPCC dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    language = 'zh'
    label_map = {'arc': {3: 'disagreeing with',
                         1: 'discussing',
                         2: 'agreeing with',
                         0: 'unrelated to'},
                 'argmin': {0: 'argument against',
                            1: 'argument for'},
                 'fnc1': {3: 'disagreeing with',
                          1: 'discussing',
                          2: 'agreeing with',
                          0: 'unrelated to'},
                 'iac1': {0: 'anti',
                          2: 'other',
                          1: 'pro'},
                 'ibmcs': {0: 'con',
                           1: 'pro'},
                 'perspectrum': {0: 'undermining',
                                 1: 'supporting'},
                 'semeval2016t6': {0: 'against',
                                   1: 'in favour',
                                   2: 'none'},
                 'snopes': {0: 'refuting',
                            1: 'agreeing with'},
                 'twitter2015': {0: 'negative',
                                 1: 'neutral',
                                 2: 'positive',
                                 3: 'UNRELATED'},
                 'twitter2017': {0: 'negative',
                                 1: 'neutral',
                                 2: 'positive'},
                 'vast': {0: 'con',
                          1: 'pro',
                          2: 'neutral',
                          3: 'UNRELATED'}
                }

    def __init__(self, dataset):
        self.dataset = dataset
        if self.dataset not in TransStanceProcessor.label_map.keys():
            raise KeyError('Dataset labelmap not found')
        self.label_map = TransStanceProcessor.label_map[self.dataset]

    def get_examples(self, data_dir, split='train'):
        """See base class."""
        examples = []
        df = pd.read_csv(os.path.join(data_dir, 'zh_version', f"{self.dataset}-{split}.csv"), sep=',', header=0)
        for (i, line) in df.iterrows():
            guid = "%s-%s" % (split, i)
            topic = line['前提']
            text = line['假设']
            label = self.label_map[line['标签']]
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
        return list(self.label_map.values())  # changing labels require overwriting cache


nlpcc_processors = {
    "stance": TransStanceProcessor,
}

nlpcc_output_modes = {
    "stance": "classification",
}

nlpcc_tasks_num_labels = {
    "stance": 4,
}

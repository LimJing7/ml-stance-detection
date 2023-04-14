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
""" Wiki Cont utils (dataset loading and evaluation) """


import logging
import numpy as np
import os
import pandas as pd

from transformers import DataProcessor
from .utils import StanceExample

logger = logging.getLogger(__name__)


class MajTwitterWikiContHighFiltlProcessor(DataProcessor):
    """Processor for the wiki controversial dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    label_map = {'AGAINST': 'against',
                 'FAVOR': 'in favour',
                 'NONE': 'neutral',
                 'DISCUSS': 'discuss'}
    language = 'en'
    contexts_dict = {}

    def __init__(self, count, random_seed):
        self.count = count
        self.random_seed = random_seed

    def get_examples(self, data_dir, split='train'):
        """See base class."""
        examples = []
        df = pd.read_csv(os.path.join(data_dir, f"majority_voted_twitter_wiki_cont_high_filtered-{split}.csv"), lineterminator='\n')
        bg = np.random.default_rng(self.random_seed).bit_generator
        if self.count == -1:
            sample = df
        else:
            sample = df.sample(n=self.count, replace=False, random_state=bg)

        for i, line in sample.iterrows():
            guid = f"{split}_{i}"
            topic = line['topic']
            text = line['text']
            # context = TwitterWikiContProcessor.contexts_dict[topic]
            if split == 'test':
                label = "neutral"
            else:
                label = str(line['pred'].strip())
            try:
                assert isinstance(topic, str) and isinstance(text, str) and isinstance(label, str)
            except AssertionError:
                print(f'index: {line[0]}')
                continue
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
        return ["against", "in favour", "neutral", "discuss"]  # changing labels require overwriting cache


maj_wiki_cont_filt_processors = {
    "stance": MajTwitterWikiContHighFiltlProcessor,
}

maj_wiki_cont_filt_output_modes = {
    "stance": "classification",
}

maj_wiki_cont_filt_tasks_num_labels = {
    "stance": 4,
}

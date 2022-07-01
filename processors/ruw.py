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
""" RUW utils (dataset loading and evaluation) """


import pickle
import logging
import os

from transformers import DataProcessor
from .utils import InputExample

logger = logging.getLogger(__name__)


class RuwProcessor(DataProcessor):
    """Processor for the ruw dataset.
    data from https://github.com/Meituan-Dianping/asap
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    language = 'zh'

    def __init__(self):
        pass

    def get_examples(self, data_dir, split='train'):
        """See base class."""
        examples = []
        data = pickle.load(open(os.path.join(data_dir, f"ruw-{split}.pkl"), 'rb'))

        guid = 0
        for user, post in data.items():
            for comment_id, comments in post.items():
                for comment in comments['data']['data']:
                    text = comment['text']
                    examples.append(InputExample(guid=guid, text_a=text))
                    guid += 1

        return examples

    def get_train_examples(self, data_dir):
        return self.get_examples(data_dir, split='train')

    def get_dev_examples(self, data_dir):
        return self.get_examples(data_dir, split='dev')

    def get_test_examples(self, data_dir, ):
        return self.get_examples(data_dir, split='test')

    def get_labels(self):
        """See base class."""
        return []


ruw_processors = {
    "stance": RuwProcessor,
}

ruw_output_modes = {
    "stance": "classification",
}

ruw_tasks_num_labels = {
    "stance": 0,
}

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


class MajTwitterTfIdfProcessor(DataProcessor):
    """Processor for the NLPCC dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    label_map = {'AGAINST': 'against',
                 'FAVOR': 'in favour',
                 'NONE': 'neutral',
                 'DISCUSS': 'discuss'}
    language = 'en'

    contexts_dict = {'ban': '',
                     'child': '',
                     'chinese': '',
                     'electric': '',
                     'firecrackers': '',
                     'iphone se': '',
                     'motorcycles': '',
                     'russia': '',
                     'second child': '',
                     'shenzhen': '',
                     'spring festival': ''}

    def __init__(self, shots=-1, random_seed=43):
        self.shots = shots
        self.random_seed = random_seed
        pass

    def get_examples(self, data_dir, split='train'):
        """See base class."""
        examples = []
        df = pd.read_csv(os.path.join(data_dir, "maj_twitter_tfidf-{}.csv".format(split)), lineterminator='\n')

        if self.shots != -1:
            for topic in MajTwitterTfIdfProcessor.contexts_dict:
                topic_df = df[df['topic']==topic]
                subset = topic_df.sample(n=min(self.shots, len(topic_df)), replace=False, random_state=self.random_seed)
                for (i, line) in enumerate(subset.iterrows()):
                    guid = "%s-%s" % (split, i)
                    topic = line[1]['topic']
                    text = line[1]['text']
                    # context = MajTwitterTfIdfProcessor.contexts_dict[topic]
                    label = str(line[1]['pred'].strip())
                    assert isinstance(topic, str) and isinstance(text, str) and isinstance(label, str)
                    examples.append(StanceExample(guid=guid, topic=topic, text=text, label=label))

        else:
            for (i, line) in enumerate(df.iterrows()):
                guid = "%s-%s" % (split, i)
                topic = line[1]['topic']
                text = line[1]['text']
                # context = MajTwitterTfIdfProcessor.contexts_dict[topic]
                label = str(line[1]['pred'].strip())
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
        return ["against", "in favour", "neutral", "discuss"]  # changing labels require overwriting cache


nlpcc_processors = {
    "stance": MajTwitterTfIdfProcessor,
}

nlpcc_output_modes = {
    "stance": "classification",
}

nlpcc_tasks_num_labels = {
    "stance": 4,
}

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


class MajTwitterBulkProcessor(DataProcessor):
    """Processor for the NLPCC dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    label_map = {'AGAINST': 'against',
                 'FAVOR': 'in favour',
                 'NONE': 'neutral',
                 'DISCUSS': 'discuss'}
    language = 'en'

    contexts_dict = {'china': 'China is the world most populous country; China is currently governed as a one party socialist republic by the CCP',
                     'sport': 'Sport pertains to any form of competitive physical activity or game that aims to use, maintain, or improve physical ability and skills while providing enjoyment to participants and, in some cases, entertainment to spectators.',
                     'apple': 'Apple Inc. is an American multinational technology company which designs, manufactures, and markets smartphones, tablets, personal computers (PCs), portable and wearable devices as well as related software and services',
                     'news': 'News is information about current events; common topics for news reports include war, government, politics, education, health, the environment, economy, business, fashion, entertainment, and sport, as well as quirky or unusual events',
                     'football': 'Football is a family of team sports that involve, to varying degrees, kicking a ball to score a goal; sports commonly called football include association football, gridiron football, Australian rules football, rugby union and rugby league, and Gaelic football.',
                     'banks': 'A bank is a financial institution that accepts deposits from the public and creates a demand deposit while simultaneously making loans.',
                     'america': 'The United States of America (U.S.A. or USA), commonly known as the United States (U.S. or US) or informally America, is a country in North America.',
                     'usa': 'The United States of America (U.S.A. or USA), commonly known as the United States (U.S. or US) or informally America, is a country in North America.',
                     'india': 'India is a country in South Asia; it is the seventh-largest country by area, the second-most populous country, and the most populous democracy in the world',
                     'EU': 'The European Union (EU) is a supranational political and economic union of 27 member states that are located primarily in Europe',
                     'tech': 'Technology is the application of knowledge to reach practical goals in a specifiable and reproducible way or the products of such an endeavor; mainly refers electronics, software, computers and information technology.',
                     'global warming': 'In common usage, climate change describes global warming; climate change has been strongly affected by climate change denial and misinformation.',
                     'fires': 'Fire is the rapid oxidation of a material (the fuel) in the exothermic chemical process of combustion, releasing heat, light, and various reaction products; fires can also refer to forest fires which are unplanned, uncontrolled and unpredictable fire in an area of combustible vegetation starting in rural and urban areas.'}

    def __init__(self, shots=-1, random_seed=43):
        self.shots = shots
        self.random_seed = random_seed
        pass

    def get_examples(self, data_dir, split='train'):
        """See base class."""
        examples = []
        df = pd.read_csv(os.path.join(data_dir, "maj_twitter_bulk-{}.csv".format(split)), lineterminator='\n')

        if self.shots != -1:
            for topic in MajTwitterBulkProcessor.contexts_dict:
                subset = df[df['topic']==topic].sample(n=self.shots, replace=False, random_state=self.random_seed)
                for (i, line) in enumerate(subset.iterrows()):
                    guid = "%s-%s" % (split, i)
                    topic = line[1]['topic']
                    text = line[1]['text']
                    context = MajTwitterBulkProcessor.contexts_dict[topic]
                    label = str(line[1]['pred'].strip())
                    assert isinstance(topic, str) and isinstance(text, str) and isinstance(label, str) and isinstance(context, str)
                    examples.append(StanceExample(guid=guid, topic=topic, text=text, label=label, context=context))

        else:
            for (i, line) in enumerate(df.iterrows()):
                guid = "%s-%s" % (split, i)
                topic = line[1]['topic']
                text = line[1]['text']
                context = MajTwitterBulkProcessor.contexts_dict[topic]
                label = str(line[1]['pred'].strip())
                assert isinstance(topic, str) and isinstance(text, str) and isinstance(label, str) and isinstance(context, str)
                examples.append(StanceExample(guid=guid, topic=topic, text=text, label=label, context=context))
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
    "stance": MajTwitterBulkProcessor,
}

nlpcc_output_modes = {
    "stance": "classification",
}

nlpcc_tasks_num_labels = {
    "stance": 4,
}

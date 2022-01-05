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
import os

from transformers import DataProcessor
from .utils import StanceExample

logger = logging.getLogger(__name__)


class NLPCCProcessor(DataProcessor):
    """Processor for the NLPCC dataset.
    Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

    def __init__(self):
        pass

    def get_examples(self, data_dir, split='train'):
      """See base class."""
      examples = []
      lines = self._read_tsv(os.path.join(data_dir, "{}.tsv".format(split)), quotechar='"')

      for (i, line) in enumerate(lines):
        guid = "%s-%s" % (split, i)
        topic = line[1]
        text = line[2]
        if split == 'test' and len(line) != 4:
          label = "UNRELATED"
        else:
          label = str(line[3].strip())
          if label == 'NONE':
            label = 'UNRELATED'
        assert isinstance(topic, str) and isinstance(text, str) and isinstance(label, str)
        examples.append(StanceExample(guid=guid, text_a=topic, text_b=text, label=label))
      return examples

    def get_train_examples(self, data_dir):
        return self.get_examples(data_dir, split='train')

    def get_dev_examples(self, data_dir):
        return self.get_examples(data_dir, split='dev')

    def get_test_examples(self, data_dir, ):
        return self.get_examples(data_dir, split='test')

    def get_labels(self):
        """See base class."""
        return ["AGAINST", "DISCUSS", "FAVOR", "UNRELATED"]


nlpcc_processors = {
    "stance": NLPCCProcessor,
}

nlpcc_output_modes = {
    "stance": "classification",
}

nlpcc_tasks_num_labels = {
    "stance": 3,
}

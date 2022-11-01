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

    label_map = {'AGAINST': 'against',
                 'FAVOR': 'in favour',
                 'NONE': 'neutral'}
    language = 'zh'
    contexts_dict = {'IphoneSE': 'iPhone SE是苹果公司在2016年3月21日10点在美国加州库比蒂诺总部发布的电子产品。',
                     '俄罗斯在叙利亚的反恐行动': '俄罗斯在叙利亚内战的军事介入指俄罗斯在叙利亚内战中采取的军事行动,俄军从2015年9月30日开始向叙利亚反政府武装展开空袭。在此之前俄罗斯已经向叙利亚政府提供武器,俄罗斯的军事行动也得到叙利亚政府同意。',
                     '开放二胎': '2015年10月26日至29日,中国共产党第十八届中央委员会第五次全体会议提出坚持计划生育的基本国策,完善人口发展战略,“全面实施一对夫妇可生育两个孩子政策”。中国从1980年开始,推行了35年的城镇人口独生子女政策真正宣告终结。',
                     '春节放鞭炮': '放鞭炮是中国传统民间习俗,但是因为担忧雾霾、噪声扰城,已有近700城市禁放、限放鞭炮',
                     '深圳禁摩限电': '2016年3月22日起到6月底,在全市开展禁摩限电源头治理行动。地铁口、公交站点、口岸、商业繁华区等区域聚集非法拉客的,一律拘留;违规使用电动三轮车的,4月1日起一律拘留。'}

    def __init__(self):
        pass

    def get_examples(self, data_dir, split='train'):
      """See base class."""
      examples = []
      lines = self._read_tsv(os.path.join(data_dir, "nlpcc-{}.tsv".format(split)), quotechar='"')

      for (i, line) in enumerate(lines):
        guid = "%s-%s" % (split, i)
        topic = line[1]
        text = line[2]
        context = NLPCCProcessor.contexts_dict[topic]
        if split == 'test' and len(line) != 4:
          label = "neutral"
        else:
          label = NLPCCProcessor.label_map[str(line[3].strip())]
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
        return ["against", "in favour", "neutral"]  # changing labels require overwriting cache


nlpcc_processors = {
    "stance": NLPCCProcessor,
}

nlpcc_output_modes = {
    "stance": "classification",
}

nlpcc_tasks_num_labels = {
    "stance": 4,
}

import json
import numpy as np

import logging
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from processors.utils import StanceExample

rng = np.random.default_rng(42)
LOG = logging.getLogger(__name__)

def perturb(examples, n_perturbations, neighbors_file):
    """Perturb the examples for robustness training

    Args:
            examples (list[StanceExamples]): List of examples to perturb
            n_perturbations (int): Number of perturbations to generate per example
            neighbors_file (str): File containing synonyms
    """

    p_examples = []

    with open(neighbors_file, 'r') as f:
        neighbors = json.load(f)

    successes = 0
    for example in tqdm.tqdm(examples):
        topic = example.topic
        text = example.text
        topic_words = topic.split(' ')
        text_words = text.split(' ')

        p_topic = []
        p_text = []

        for i in range(n_perturbations):
            success = False
            p_topic_words = []
            p_text_words = []
            for word in topic_words:
                if word not in neighbors:
                    p_topic_words.append(word)
                elif len(neighbors[word]) == 0:
                    p_topic_words.append(word)
                else:
                    success = True
                    tmp_list = [word] + neighbors[word]
                    idx = rng.integers(0, len(tmp_list))
                    p_topic_words.append(tmp_list[idx])
            for word in text_words:
                if word not in neighbors:
                    p_text_words.append(word)
                elif len(neighbors[word]) == 0:
                    p_text_words.append(word)
                else:
                    success = True
                    tmp_list = [word] + neighbors[word]
                    idx = rng.integers(0, len(tmp_list))
                    p_text_words.append(tmp_list[idx])

            assert len(p_topic_words) == len(topic_words)
            assert len(p_text_words) == len(text_words)

            p_topic = ' '.join(p_topic_words)
            p_text = ' '.join(p_text_words)

            p_example = StanceExample(guid=example.guid, topic=p_topic, text=p_text, label=example.label)
            p_examples.append(p_example)

            if success:
                successes += 1

    with logging_redirect_tqdm():
        LOG.info(f'{successes}/{n_perturbations * len(examples)} = {successes/(n_perturbations * len(examples))}')

    return p_examples

from collections import defaultdict
from nltk.corpus import wordnet
import pickle

from run_classify import PROCESSORS

def get_synonyms(word):
    synonyms = []
    synsets = wordnet.synsets(word)
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return synonyms


all_synonyms = defaultdict(dict)
for ds, processor in PROCESSORS['stance'].items():
    for label in processor().get_labels():
        synonyms = get_synonyms(label)
        all_synonyms[ds][label] = synonyms

with open('synonyms.pkl', 'wb') as f:
    pickle.dump(all_synonyms, f)

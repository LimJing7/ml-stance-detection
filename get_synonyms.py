from bs4 import BeautifulSoup
from collections import defaultdict
from nltk.corpus import wordnet
import pickle
import requests

from run_classify import PROCESSORS

def get_wordnet_synonyms(word):
    synonyms = []
    synsets = wordnet.synsets(word)
    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def get_wiki_synonyms(word):
    syns = []

    session = requests.Session()
    url = f"https://en.wiktionary.org/wiki/Thesaurus:{word}"
    response = session.get(url)
    soup = BeautifulSoup(response.text.replace('>\n<', '><'), 'html.parser')
    lala = soup.find_all('span', {'id': 'Synonyms'})

    contents = soup.find_all('span', {'class': 'toctext'})
    for content in contents:
        if content.text == 'Synonyms':
            content_index = content.find_previous().text
            content_id = content.parent['href'].replace('#', '')
            span_tag = soup.find_all('span', {'id': content_id})[0]

            parent_tag = span_tag.parent
            while not parent_tag.find_all('li'):
                parent_tag = parent_tag.find_next_sibling()
            for list_tag in parent_tag.find_all('li'):
                syns.append(list_tag.text)

    return syns

label_map = {'disagreeing with': 'disagree',
             'agreeing with': 'agree',
             'unrelated to': 'unrelated',
             'argument against': 'against',
             'argument for': 'for',
             'in favour': 'favour',
             'contradiction of': 'contradiction',
             'paraphrase of': 'paraphrase'
             }


all_synonyms = defaultdict(dict)
for ds, processor in PROCESSORS['stance'].items():
    for label in processor().get_labels():
        if len(label.split(' ')) > 1:
            label = label_map[label]
        wordnet_synonyms = get_wordnet_synonyms(label)
        wiki_synonyms = get_wiki_synonyms(label)
        synonyms = set(wordnet_synonyms + wiki_synonyms)
        all_synonyms[ds][label] = synonyms

with open('synonyms.pkl', 'wb') as f:
    pickle.dump(all_synonyms, f)

import argparse
from collections import defaultdict
import copy
import jsonlines
import random

target_frac = {'unrel': 0.6,
               'pos': 0.07,
               'neg': 0.1,
               'neu': 0.23}

def check_frac(current, target):
    for key in current:
        if abs(current[key] - target[key]) > 3:
            return key
    return True

def rebase(frac_output, unrel, pos, neu, neg):
    sizes = {'unrel': len(unrel),
             'pos': len(pos),
             'neu': len(neu),
             'neg': len(neg)}
    base = sizes[frac_output] / target_frac[frac_output]
    target_unrel = target_frac['unrel'] * base
    unrel = unrel[:target_unrel]
    target_pos = target_frac['pos'] * base
    pos = pos[:target_pos]
    target_neu = target_frac['neu'] * base
    neu = neu[:target_neu]
    target_neg = target_frac['neg'] * base
    neg = neg[:target_neg]

    return unrel, pos, neu, neg


def convert(file):
    data = []
    titles = defaultdict(list)
    with jsonlines.open(file) as f:
        for line in f:
            data.append(line)
            titles[line['lang']].append((line['title'], line['heading']))
    
    for k in titles:
        random.shuffle(titles[k])

    lang_counter = defaultdict(int)
    new_data = []
    for example in data:
        new_title, new_heading = titles[example['lang']][lang_counter[example['lang']]]
        if example['title'] != new_title and example['heading'] != new_heading:
            new_example = copy.copy(example)
            new_example['title'] = new_title
            new_example['heading'] = new_heading
            new_example['c_xlmr_senti'] = 'unrelated'
            new_data.append(new_example)
        lang_counter[example['lang']] += 1
    
    unrel = new_data
    pos = [i for i in data if i['c_xlmr_senti'] == 'Positive']
    neu = [i for i in data if i['c_xlmr_senti'] == 'Neutral']
    neg = [i for i in data if i['c_xlmr_senti'] == 'Negative']

    total = len(data) + len(new_data)
    current_frac = {
        'unrel': len(unrel) / total,
        'pos': len(pos) / total,
        'neu': len(neu) / total,
        'neg': len(neg) / total,
    }

    frac_output = check_frac(current_frac, target_frac)
    while frac_output is not True:
        unrel, pos, neu, neg = rebase(frac_output, unrel, pos, neu, neg)
    
    all_data = unrel + pos + neu + neg
    print(f'{len(all_data)=}')
    return all_data

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', required=True, help='data directory')
    parser.add_argument('--output_dir', required=True, help='output directory')
    parser.add_argument('--seed', type=int, required=False, default=43, help='random seed')

    args = parser.parse_args()

    random.seed(args.seed)

    en_train = convert(args.data_dir + '/english/enwiki_train.json')
    with open(args.output_dir + '/sentistance-enwiki-train.json', 'w') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(en_train)
    en_dev = convert(args.data_dir + '/english/enwiki_dev.json')
    with open(args.output_dir + '/sentistance-enwiki-dev.json', 'w') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(en_dev)
    en_test = convert(args.data_dir + '/english/enwiki_test.json')
    with open(args.output_dir + '/sentistance-enwiki-test.json', 'w') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(en_test)
        
    mul_train = convert(args.data_dir + '/multilingual/mwiki_train.json')
    with open(args.output_dir + '/sentistance-mwiki-train.json', 'w') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(mul_train)
    mul_dev = convert(args.data_dir + '/multilingual/mwiki_dev.json')
    with open(args.output_dir + '/sentistance-mwiki-dev.json', 'w') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(mul_dev)
    mul_test = convert(args.data_dir + '/multilingual/mwiki_test.json')
    with open(args.output_dir + '/sentistance-mwiki-test.json', 'w') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(mul_test)



if __name__ == '__main__':
    main()
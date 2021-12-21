import argparse
from collections import defaultdict
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True,
                        help='file to split')
    parser.add_argument('--random_seed', type=int, required=False, default=0,
                        help='random seed')

    args = parser.parse_args()

    random.seed(args.random_seed)

    with open(args.file, 'r', newline='\r\n', encoding='utf-8-sig') as f:
        header = f.readline().strip().split('\t')
        data = []
        for line in f:
            data.append(line.strip().split('\t'))

    data_by_topic = defaultdict(list)
    train = []
    dev = []
    test = []

    for row in data:
        topic = row[1]
        data_by_topic[topic].append(row)

    for topic, contexts in data_by_topic.items():
        random.shuffle(contexts)
        n_context = len(contexts)
        train.extend(contexts[:int(0.8*n_context)])
        dev.extend(contexts[int(0.8*n_context):int(0.9*n_context)])
        test.extend(contexts[int(0.9*n_context):])

    fname_wo_ext = '.'.join(args.file.split('.')[:-1])

    with open(f'{fname_wo_ext}-train.tsv', 'w') as f:
        f.write('"'+'"\t"'.join(header)+'"\n')
        for row in train:
            f.write('"'+'"\t"'.join(row)+'"\n')
    with open(f'{fname_wo_ext}-dev.tsv', 'w') as f:
        f.write('"'+'"\t"'.join(header)+'"\n')
        for row in dev:
            f.write('"'+'"\t"'.join(row)+'"\n')
    with open(f'{fname_wo_ext}-test.tsv', 'w') as f:
        f.write('"'+'"\t"'.join(header)+'"\n')
        for row in test:
            f.write('"'+'"\t"'.join(row)+'"\n')


if __name__ == '__main__':
    main()

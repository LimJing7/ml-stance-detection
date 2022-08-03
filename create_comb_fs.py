import argparse
import csv
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_count', type=int, help='number of train examples')
    parser.add_argument('--dir', default='/projects/tir5/users/jinglim/stance_datasets', help='dir of dataset')
    parser.add_argument('--seed', default=42, type=int, help='random seed')

    args = parser.parse_args()

    train_set = []
    with open(f'{args.dir}/nlpcc-train.tsv', 'r') as train_file:
        reader = csv.reader(train_file, delimiter='\t')
        for line in reader:
            train_set.append(line)

    dev_set = []
    with open(f'{args.dir}/nlpcc-dev.tsv', 'r') as dev_file:
        reader = csv.reader(dev_file, delimiter='\t')
        for line in reader:
            dev_set.append(line)

    random.seed(args.seed)
    random.shuffle(train_set)

    fewshot_train = train_set[:args.train_count]
    comb_dev = dev_set + train_set[args.train_count:]

    with open(f'{args.dir}/comb_nlpcc_{args.train_count}-train.tsv', 'w') as write_file:
        writer = csv.writer(write_file, delimiter='\t')
        writer.writerows(fewshot_train)

    with open(f'{args.dir}/comb_nlpcc_{args.train_count}-dev.tsv', 'w') as write_file:
        writer = csv.writer(write_file, delimiter='\t')
        writer.writerows(comb_dev)

    print(f'{len(fewshot_train)=}')
    print(f'{len(comb_dev)=}')

if __name__ == '__main__':
    main()
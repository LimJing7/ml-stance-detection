import argparse
import csv
import random
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True,
                        help='file to split')
    parser.add_argument('--random_seed', type=int, required=False, default=0,
                        help='random seed')

    args = parser.parse_args()

    random.seed(args.random_seed)

    df = pd.read_csv(args.file)

    test = df[df['Set'] == 'Test']
    traindev = df[df['Set'] == 'Training']

    train = traindev.sample(frac=0.75, replace=False, random_state=args.random_seed)
    dev = traindev.drop(train.index)

    fdir = '/'.join(args.file.split('/')[:-1])

    train.to_csv(f'{fdir}/train.csv', index=False, header=True, sep=',', quoting=csv.QUOTE_ALL)
    dev.to_csv(f'{fdir}/dev.csv', index=False, header=True, sep=',', quoting=csv.QUOTE_ALL)
    test.to_csv(f'{fdir}/test.csv', index=False, header=True, sep=',', quoting=csv.QUOTE_ALL)

if __name__ == '__main__':
    main()

import argparse
import csv
import random
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--macron_file', type=str, required=True,
                        help='file to split')
    parser.add_argument('--lepen_file', type=str, required=True,
                        help='file to split')
    parser.add_argument('--random_seed', type=int, required=False, default=0,
                        help='random seed')

    args = parser.parse_args()

    random.seed(args.random_seed)

    macron_df = pd.read_csv(args.macron_file)
    macron_df['Target'] = 'Emmanuel Macron'
    lepen_df = pd.read_csv(args.lepen_file)
    lepen_df['Target'] = 'Marine Le Pen'

    df = pd.concat([macron_df, lepen_df], axis=0, ignore_index=True)

    test = df[df['Set'] == 'Test']
    traindev = df[df['Set'] == 'Training']

    train = traindev.sample(frac=0.75, replace=False, random_state=args.random_seed)
    dev = traindev.drop(train.index)

    fdir = '/'.join(args.macron_file.split('/')[:-1])

    train.to_csv(f'{fdir}/train.csv', index=False, header=True, sep=',', quoting=csv.QUOTE_ALL)
    dev.to_csv(f'{fdir}/dev.csv', index=False, header=True, sep=',', quoting=csv.QUOTE_ALL)
    test.to_csv(f'{fdir}/test.csv', index=False, header=True, sep=',', quoting=csv.QUOTE_ALL)

if __name__ == '__main__':
    main()

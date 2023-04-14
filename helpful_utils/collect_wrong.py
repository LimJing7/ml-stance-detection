import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--filename', required=True, help='filename')
parser.add_argument('--output', required=True, help='output filename')

args = parser.parse_args()

df = pd.read_csv(args.filename, sep='\t', lineterminator='\n', names=['pred', 'gold', 'text'])
wrong_df = df[df['pred'] != df['gold']]

wrong_df.to_csv(args.output, sep=',', index=False)
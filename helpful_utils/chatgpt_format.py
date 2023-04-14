import pandas as pd

#df = pd.read_csv('../stance_datasets/comb_nlpcc-dev.tsv', sep='\t', lineterminator='\n', header=None)
# df.columns = ['id', 'topic', 'text', 'label']
df = pd.read_csv('./my_helpful_wrong_split', lineterminator='\n')

for i, row in df.iterrows():
    output = f"{i+1}: Which of the three choices, 'in favor', 'against', 'neutral' is the closest to the stance of the author of {row['text']} towards {row['topic']}?"
    print(f'gold: {row["gold"]}')
    print(f'pred: {row["pred"]}')
    print(output)
    inp = input('Press any key to continue')

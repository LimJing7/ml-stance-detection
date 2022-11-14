import pandas as pd

df = pd.read_csv('deduped_twitter_efra_en.csv', lineterminator='\n')

to_keep = []

## flip the order to keep oldest iteration
for idx, row in df.iterrows():
    if 'macron' in str(row['text']).lower():
        to_keep.append(True)
    else:

        to_keep.append(False)

filtered_df = df[to_keep]

filtered_df.to_csv('deduped_twitter_efra_macron_en.csv')
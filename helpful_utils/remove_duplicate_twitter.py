import pandas as pd

df = pd.read_csv('twitter_iphonese_en.csv', lineterminator='\n')

existing = set()
to_keep = []

## flip the order to keep oldest iteration
for idx, row in df[::-1].iterrows():
    if row['text'] in existing:
        to_keep.append(False)
    else:
        to_keep.append(True)
        existing.add(row['text'])

filtered_df = df[::-1][to_keep]

filtered_df.to_csv('deduped_twitter_iphonese_en.csv')
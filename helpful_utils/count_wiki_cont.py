import pandas as pd

low_threshold = 0.1
high_threshold = 0.8
dfs = []

for part in range(12):
    print(f'{part = }')
    df = pd.read_csv(f'./twitter_data/majority_voted_twitter_wiki_cont_high_part_{part}.csv', lineterminator='\n')
    dfs.append(df)

dataset = pd.concat(dfs, ignore_index=True)

counts = dataset.groupby(['topic', 'pred']).count()
per_topic_counts = dataset.groupby(['topic']).count().drop('pred', axis=1)
proportion = counts/per_topic_counts

neutral_proportion = proportion.xs('neutral', level='pred')
against_proportion = proportion.xs('against', level='pred')
in_favour_proportion = proportion.xs('in favour', level='pred')

high_neu_pro = neutral_proportion['text'] > low_threshold
low_neu_pro = neutral_proportion['text'] < high_threshold
sel_neu_pro = high_neu_pro * low_neu_pro
sel_neu_topics = neutral_proportion[sel_neu_pro]

high_fav_pro = in_favour_proportion['text'] > low_threshold
low_fav_pro = in_favour_proportion['text'] < high_threshold
sel_fav_pro = high_fav_pro * low_fav_pro
sel_fav_topics = in_favour_proportion[sel_fav_pro]

high_aga_pro = against_proportion['text'] > low_threshold
low_aga_pro = against_proportion['text'] < high_threshold
sel_aga_pro = high_aga_pro * low_aga_pro
sel_aga_topics = against_proportion[sel_aga_pro]

sel_topics = set(sel_neu_topics.index).intersection(set(sel_fav_topics.index), set(sel_aga_topics.index))

counts.to_csv(f'twitter_data/majority_voted_twitter_wiki_cont_high_counts.csv')
filtered_dataset = dataset[dataset['topic'].isin(sel_topics)]
filtered_dataset.to_csv(f'twitter_data/majority_voted_twitter_wiki_cont_high_filtered.csv', index=False)

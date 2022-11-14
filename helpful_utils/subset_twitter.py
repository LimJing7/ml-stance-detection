import pandas as pd

df = pd.read_csv('majority_voted_twitter_efra.csv')
final_df = df.sample(n=70000, random_state=42)

final_df.to_csv('subset_majority_voted_twitter_efra.csv')
print(final_df.shape)
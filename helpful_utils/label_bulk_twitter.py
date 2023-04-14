import pandas as pd
import tqdm

first_tweets = {'china': 'RT @Guangming_Daily: Herdsmen tame horses on snow-covered pasture in N China   ',
                'sport': 'Man Utd to get £1m from Palace if they return Fosu-Mensah  ',
                'apple': 'RT @JoelOsteen: We all go through seasons of struggle, but that’s not your permanent home. You’re coming into an anointing of ease. Stay en…',
                'news': 'They favor a lot.  ',
                'football': 'Man Utd to get £1m from Palace if they return Fosu-Mensah  ',
                'banks': 'RT @Banks: @KEEMSTAR @Baited_Podcast Love you three, need more fun in my life. Excited.',
                'america': 'RT @peterbakernyt: Russian submarines have dramatically stepped up activity around undersea cables that provide Internet and other communic…',
                'usa': 'K5BL LIVE Weather McKinney TEXAS USA ... 6:00 PM Temperature  35.5oF Humidity 83 PCT % Wind 0.0 mph N%',
                'india': 'RT @PSUwMlQ6ooiSg6w: @kabirisGodLord Sam Darnold\n#ThingsNeverSaidIn2017\nProf. Cheiro : No power of the world can stop India from radiating…',
                'EU': "RT @taevinyls: before 2017 ends i want to clear a few things out:\n\n• jin is not a pink 'princess'\n• nor a 'mom'\n• SHIP WARS ARE UNNECESSARY…",
                'tech': 'RT @charlesmilander: What telecoms and tech companies are saying about the FCC’s net neutrality decision\n from 0-100…',
                'global warming': "RT @ClimateReality: Mr. President, we recommend downloading our free e-book. In it, we explain why global warming doesn't mean the absence…",
                'fires': "RT @PLeonardNYDN: Bobby Hart is confirming on Twitter that he quit. You can't make the Giants' 2017 season up. And this was a roster Jerry…"}

search_terms = ['china', 'sport', 'apple', 'news', 'football', 'banks', 'america', 'usa', 'india', 'EU', 'tech', 'global warming', 'fires']

df = pd.read_csv('twitter_bulk.csv')
df['topic'] = ''

c_search_ID = 0
for i, row in tqdm.tqdm(df.iterrows()):
    try:
        if row['text'] == first_tweets[search_terms[c_search_ID]]:
            c_search_ID += 1
    except IndexError:
        pass
    row['topic'] = search_terms[c_search_ID-1]

df.to_csv('twitter_bulk_w_topics.csv')
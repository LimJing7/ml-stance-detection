import datetime
import json
import pandas as pd
import requests
import time

import twitter_secrets

url = 'https://api.twitter.com/2/tweets/search/all'

# +- 2 wks of the election; range copied from paper
params = {}

start_date = datetime.datetime(2017,4,20,tzinfo=datetime.timezone(datetime.timedelta(0)))
end_date = datetime.datetime(2017,5,20,tzinfo=datetime.timezone(datetime.timedelta(0)))
query = '(macron OR #presidentielles2017 OR lepen OR le pen) lang:en -is:retweet'
max_results = 500
tweet_field = 'created_at,lang,entities'

params['start_time'] = start_date.isoformat()
params['end_time'] = end_date.isoformat()
params['query'] = query
params['max_results'] = max_results
params['tweet.fields'] = tweet_field


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {twitter_secrets.acad_bearer_token}"
    r.headers["User-Agent"] = "twitter_acad"
    return r


collected_text = []
final = False

# ### continue from crash
# next_token = '1jzu9lk96gu5npw3jg1adufdsa1adrzquipxd510s48t'
# with open('twitter_efra_wip.json', 'r') as f:
#     collected_text = json.load(f)

while not final:
    re = requests.get(url=url, auth=bearer_oauth, params=params)
    if re.status_code == 429:
        with open('twitter_efra.json', 'w') as f:
            json.dump(collected_text, f)
        print(f'{next_token = }')
        time.sleep(15*60)
        re = requests.get(url=url, auth=bearer_oauth, params=params)
    elif re.status_code != 200:
        with open('twitter_efra_wip.json', 'w') as f:
            json.dump(collected_text, f)
        print(f'{next_token = }')
        raise Exception(re.status_code, re.text)
    else:
        meta = re.json()['meta']
        try:
            next_token = meta['next_token']
            params['next_token'] = next_token
            print(f'{next_token = }')
        except KeyError:
            final = True

        # for entry in re.json()['data']:
        #     if entry['lang'] != 'en':
        #         collected_other_lang.append(entry['text'])
        #     else:
        #         collected_text.append(entry['text'])
        collected_text += re.json()['data']

# strip urls
for tweet in collected_text:
    urls_idx = []
    try:
        for url in tweet['entities']['urls']:
            s = url['start']
            e = url['end']
            urls_idx.append((s,e))
    except KeyError:
        pass
    urls_idx = sorted(urls_idx, reverse=True)
    text = tweet['text']
    for s,e in urls_idx:
        text = text[:s] + text[e:]
    tweet['text'] = text

df = pd.DataFrame(collected_text)

df.to_csv('twitter_efra.csv')

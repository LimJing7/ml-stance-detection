import datetime
import json
import pandas as pd
import requests
import time

import twitter_secrets

url = 'https://api.twitter.com/2/tweets/search/all'

#2015-2017
params = {}

start_date = datetime.datetime(2015,1,1,tzinfo=datetime.timezone(datetime.timedelta(0)))
end_date = datetime.datetime(2017,12,31,tzinfo=datetime.timezone(datetime.timedelta(0)))
query = '#IphoneSE lang:en'
max_results = 500
tweet_field = 'created_at,lang'

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

while not final:
    re = requests.get(url=url, auth=bearer_oauth, params=params)
    if re.status_code == 429:
        with open('twitter_iphonese_wip.json', 'w') as f:
            json.dump(collected_text, f)
        print(f'{next_token = }')
        time.sleep(15*60)
        re = requests.get(url=url, auth=bearer_oauth, params=params)
    elif re.status_code != 200:
        with open('twitter_iphonese_wip.json', 'w') as f:
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

df = pd.DataFrame(collected_text)

df.to_csv('twitter_iphonese_en.csv')
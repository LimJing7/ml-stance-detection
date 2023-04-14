import datetime
import json
import logging
import pandas as pd
import pickle
import requests
import time

import twitter_secrets

logger = logging.getLogger(__name__)
logging.basicConfig(handlers=[logging.StreamHandler()],
                    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
url = 'https://api.twitter.com/2/tweets/search/all'

# arc argmin fnc1 iac1 ibmcs perspectrum semeval2016t6 snopes twitter2015 twitter2017 vast

naming_infix = 'wiki_cont_3b'


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {twitter_secrets.acad_bearer_token}"
    r.headers["User-Agent"] = "twitter_acad"
    return r

def get_data(search_term, nPosts, start_nxt_tokens):
    save_term = search_term.replace('/', '_')
    logger.info(f'{search_term = }')
    #2015-2022
    params = {}

    start_date = datetime.datetime(2015,1,1,tzinfo=datetime.timezone(datetime.timedelta(0)))
    end_date = datetime.datetime(2022,12,31,tzinfo=datetime.timezone(datetime.timedelta(0)))
    query = f'{search_term} lang:en -is:retweet'
    max_results = 500
    tweet_field = 'created_at,lang,entities'

    params['start_time'] = start_date.isoformat()
    params['end_time'] = end_date.isoformat()
    params['query'] = query
    params['max_results'] = max_results
    params['tweet.fields'] = tweet_field

    if search_term in start_nxt_tokens:
        next_token = start_nxt_tokens[search_term]
        if len(next_token) >= 1:
            params['next_token'] = next_token
            logger.info(f'start next_token: {next_token}')
    else:
        next_token = ''

    collected_text = []
    nText = 0
    final = False


    while not final:
        re = requests.get(url=url, auth=bearer_oauth, params=params)
        if re.status_code == 429:
            with open(f'twitter_data/stance_temp/twitter_{naming_infix}_{save_term}_wip.json', 'w') as f:
                json.dump(collected_text, f, indent=0)
            with open(f'twitter_data/stance_temp/twitter_{naming_infix}_nxt_token.json', 'w') as f:
                f.write(search_term+'\n')
                f.write(next_token+'\n')
            logger.info(f'{next_token = }')
            time.sleep(15*60)
            re = requests.get(url=url, auth=bearer_oauth, params=params)
        elif re.status_code != 200:
            logger.info(re.status_code)
            with open(f'twitter_data/stance_temp/twitter_{naming_infix}_{save_term}_wip.json', 'w') as f:
                json.dump(collected_text, f, indent=0)
            logger.info(f'{next_token = }')
            raise Exception(re.status_code, [re.headers, re.text])
        else:
            meta = re.json()['meta']
            try:
                next_token = meta['next_token']
                params['next_token'] = next_token
                logger.info(f'{next_token = }')
            except KeyError:
                final = True

            # for entry in re.json()['data']:
            #     if entry['lang'] != 'en':
            #         collected_other_lang.append(entry['text'])
            #     else:
            #         collected_text.append(entry['text'])

            if meta['result_count'] == 0:
                final = True
            else:
                req_data = re.json()['data']
                for i in req_data:
                    i['search_term'] = search_term
                collected_text += req_data
                nText += max_results

                if nText >= nPosts:
                    final = True

    with open(f'twitter_data/{naming_infix}_temp/twitter_{naming_infix}_{save_term}_wip.json', 'w') as f:
        json.dump(collected_text, f, indent=0)

    return collected_text


collected_text = []
search_terms = []
with open('controversial.txt') as f:
    for line in f:
        search_terms.append(line.strip())

start_nxt_tokens = pickle.load(open('./twitter_wiki_cont_nxt_tokens.pkl', 'rb'))

sid = search_terms.index('George Floyd')

for search_term in search_terms[sid:]:
    st = search_term.strip()
    st = f'"{st}"'
    text = get_data(st, 10000, start_nxt_tokens)
    collected_text.extend(text)

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

# # merge with original
# orig_df = pd.read_csv('./twitter_data/twitter_wiki_cont.csv',lineterminator='\n',delimiter=',',index_col=0)
# df = pd.concat([orig_df, df], ignore_index=True)

df.to_csv(f'twitter_data/twitter_{naming_infix}.csv', index=False)

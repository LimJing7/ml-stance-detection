import argparse
import pandas as pd
import pickle

checkpoint_order = ['comb_', 'xstance_de', 'rita', 'efra', 'check']

def sort_checkpt_types(checkpt_types):
    output = []
    for item in checkpoint_order:
        for checkpt_type in checkpt_types:
            if checkpt_type.startswith(item):
                output.append(checkpt_type)
                break
    return output

parser = argparse.ArgumentParser()
parser.add_argument('--file', default='/projects/tir5/users/jinglim/ml-stance-detection/big_results.pkl')
parser.add_argument('--excel_file', default='/projects/tir5/users/jinglim/ml-stance-detection/big_results.xlsx')
parser.add_argument('--start_idx', type=int, required=True, help='inclusive')
parser.add_argument('--end_idx', type=int, default=-1, help='exclusive')

args = parser.parse_args()

data_df = pickle.load(open(args.file, 'rb'))

if args.end_idx == -1:
    args.end_idx = data_df.shape[0]

locations = list(data_df.loc[args.start_idx:args.end_idx, 'dir'])
m = []

for i, loc in enumerate(locations):
    if i not in m:
        parts = loc.split('/')
        prefix = '/'.join(parts[:-1])
        checkpt_types = []
        for loc2 in locations:
            if loc2.startswith(prefix):
                checkpt_types.append(loc2.split('/')[-1])
        checkpt_types = sort_checkpt_types(checkpt_types)
        for checkpt_type in checkpt_types:
            for repeat in range(5):
                full = f'{prefix[:-1]}{repeat}/{checkpt_type}'
                m.append(locations.index(full))

m = [i+args.start_idx for i in m]
print(m)

m = list(range(args.start_idx)) + m + list(range(args.end_idx, data_df.shape[0]))

## checks
if len(set(m)) != data_df.shape[0]:
    raise ValueError(f'number of entries in m is not correct -- expected: {data_df.shape[0]}, actual: {len(set(m))}')
for i in range(data_df.shape[0]):
    if i not in m:
        raise ValueError(f'value of entries in m is not correct -- {i} is missing')

data_df = data_df.reindex(m)
data_df = pd.concat([data_df], ignore_index=True)

data_df.to_excel(args.excel_file, index=False)
pickle.dump(data_df, open(args.file, 'wb'))
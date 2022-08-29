import argparse
import numpy as np

topics = [
    'IphoneSE',
    '春节放鞭炮',
    '深圳禁摩限电',
    '俄罗斯在叙利亚的反恐行动',
    '开放二胎',
]
metrics = [
    'precisions',
    'recalls',
    'f1s'
]

parser = argparse.ArgumentParser()
parser.add_argument('--filename')

args = parser.parse_args()

collated_met = np.zeros((3,5,3))

for i in range(5):
    doc = f'{args.filename}{i}/checkpoint-best/eval_results'

    with open(doc) as f:
        for line in f:
            row = list(map(lambda x: x.strip(), line.split('=')))
            if len(row) < 2:
                continue
            metric = row[0]
            value = row[1]
            if metric.split('_')[0] in topics:
                topic_id = topics.index(metric.split('_')[0])
                metric_id = metrics.index(metric.split('_')[1])
                collated_met[metric_id, topic_id] += list(map(float, value[1:-1].split()))

average_metrics = collated_met / 5
average_metrics = np.concatenate([average_metrics, np.mean(average_metrics, 1, keepdims=True)], axis=1)
average_metrics = np.concatenate([average_metrics, np.mean(average_metrics, 2, keepdims=True)], axis=2)

print('precision:')
print(average_metrics[0])

print('recall:')
print(average_metrics[1])

print('f1:')
print(average_metrics[2])
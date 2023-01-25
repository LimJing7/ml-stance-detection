import argparse
import datetime
import os
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('--folders', nargs='+')

args = parser.parse_args()

times = []
for folder in args.folders:
    with open(os.path.join(folder, 'train')) as f:
        for line in f:
            if 'Start time: ' in line:
                x = line.find('Start time: ')
                s = datetime.datetime.fromisoformat(line[x+12:].strip())
            elif 'End time: ' in line:
                x = line.find('End time: ')
                e = datetime.datetime.fromisoformat(line[x+10:].strip())
    times.append(e-s)

print([str(i) for i in times])
total = datetime.timedelta()
count = 0
for time in times:
    total += time
    count += 1
print(total/count)
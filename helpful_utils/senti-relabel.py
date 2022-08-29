import argparse
import json
import jsonlines


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', nargs='+')
    parser.add_argument('--mapping', type=json.loads)

    args = parser.parse_args()
    print(args.mapping)

    output = []
    for fname in args.fname:
        with jsonlines.open(fname) as f:
            for line in f:
                line['label'] = args.mapping[line['label']]
                output.append(line)

        with jsonlines.open(fname, 'w') as f:
            f.write_all(output)


if __name__ == '__main__':
    main()
import argparse
import pickle
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rank the embeddings in choices by distance to embeddings in target')
    parser.add_argument('--choices_file', help='file that contains the embeddings to rank')
    parser.add_argument('--target_file', help='file that contains the target embeddings')
    parser.add_argument('--output_file')

    args = parser.parse_args()

    choices = pickle.load(open(args.choices_file, 'rb'))
    target = pickle.load(open(args.target_file, 'rb'))

    choices_key = []
    choices_val = []
    for k,v in choices.items():
        choices_key.append(k)
        choices_val.append(v)

    choices_tensor = torch.stack(list(choices_val))
    target_tensor = torch.stack(list(target.values()))

    dist_tensor = choices_tensor @ target_tensor.T
    min_tensor = torch.min(dist_tensor, dim=1).values
    sorted_indices = torch.sort(min_tensor, descending=True).indices

    output_list = [choices_key[i] for i in sorted_indices]
    with open(args.output_file, 'w') as f:
        for topic in output_list:
            f.write(topic)
            f.write('\n')
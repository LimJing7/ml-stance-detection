import argparse
import hashlib
import os
import pandas as pd
import pickle
import torch

headers = [
    'model_base',
    'train_datasets',
    'task_variant',
    'max_seq_len',
    'mlm',
    'mlm_prob',
    'alpha',
    'neg_samples',
    'ds_weights',
    'ds_alpha',
    'robust',
    'robust_size',
    'robust_samples',
    'nonsup_simcse',
    'sup_simcse',
    'simcse_temp',
    'parallel_dataset',
    'extra_mlm',
    'extra_mlm_datasets',
    'mlm_join',
    'ud',
    'ud_datasets',
    'embed_dotproduct',
    'lang_discrim',
    'ld_dataset',
    'per_gpu_train_batch_size',
    'learning_rate',
    'weight_decay',
    'adam_epsilon',
    'max_grad_norm',
    'num_train_epochs',
    'nlpcc_f1',
    'comb_nlpcc_f1',
    'tnlpcc_f1',
    'dir',
    'hash'
]


def get_result(folder, dataset, df, hashes):
    train_args = torch.load(open(os.path.join(folder, 'training_args.bin'), 'rb'))

    output_row = {
        'model_base': train_args.model_name_or_path,
        'train_datasets': train_args.train_dataset,
        'task_variant': f'{train_args.task_name}_{train_args.variant}',
        'max_seq_len': train_args.max_seq_length,
        'mlm': train_args.mlm,
        'mlm_prob': train_args.mlm_probability,
        'alpha': train_args.alpha,
        'neg_samples': train_args.negative_samples,
        'ds_weights': train_args.ds_weights,
        'ds_alpha': train_args.ds_alpha,
        'robust': train_args.robust,
        'robust_size': train_args.robust_size,
        'robust_samples': train_args.robust_samples,
        'nonsup_simcse': train_args.nonsup_simcse,
        'sup_simcse': train_args.sup_simcse,
        'simcse_temp': train_args.simcse_temp,
        'parallel_dataset': train_args.parallel_dataset,
        'extra_mlm': train_args.extra_mlm,
        'extra_mlm_datasets': train_args.mlm_dataset,
        'mlm_join': train_args.mlm_join_examples,
        'ud': train_args.ud,
        'ud_datasets': train_args.ud_datasets,
        'embed_dotproduct': train_args.use_embed_dotproduct,
        'lang_discrim': train_args.lang_discrim,
        'ld_dataset': train_args.ld_dataset,
        'per_gpu_train_batch_size': train_args.per_gpu_train_batch_size,
        'learning_rate': train_args.learning_rate,
        'weight_decay': train_args.weight_decay,
        'adam_epsilon': train_args.adam_epsilon,
        'max_grad_norm': train_args.max_grad_norm,
        'num_train_epochs': train_args.num_train_epochs,
        'dir': folder,
    }

    hash_val = hashlib.sha256(str(output_row.values()).encode()).hexdigest()
    output_row['hash'] = hash_val

    with open(os.path.join(folder, 'eval_results')) as f:
        for line in f:
            if line.startswith('macro f1'):
                f1 = float(line.strip().split('=')[1])

    var_f1 = f'{dataset}_f1'

    if hash_val in hashes:
        df.loc[df['hash']==hash_val, var_f1] = f1
        return None
    else:
        hashes.add(hash_val)
        output_row[var_f1] = f1
        return output_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folders",
        nargs="*",
        type=str,
        required=True,
        help="Train dataset(s)."
    )
    parser.add_argument(
        '--output_file',
        default='big_results.pkl',
        help='file where all the results are saved'
    )
    parser.add_argument(
        '--excel_file',
        default='big_results.xlsx',
        help='excel file of results'
    )
    parser.add_argument(
        '--eval_dataset',
        choices=['nlpcc', 'tnlpcc', 'comb_nlpcc'],
        help='which dataset\'s score',
        required=True,
    )

    args = parser.parse_args()

    try:
        data_df = pickle.load(open(args.output_file, 'rb'))
    except FileNotFoundError:
        data_df = pd.DataFrame(columns=headers)
    
    hashes = set(data_df['hash'])
    
    results = []
    for folder in args.folders:
        result_dict = get_result(folder, args.eval_dataset, data_df, hashes)
        if result_dict is not None:
            results.append(result_dict)
    results_df = pd.DataFrame(results)

    data_df = pd.concat([data_df, results_df], ignore_index=True)

    pickle.dump(data_df, open(args.output_file, 'wb'))
    data_df.to_excel(args.excel_file, index=False)

if __name__ == '__main__':
    main()
import argparse
import numpy as np
import pandas as pd
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--orig_file', required=True)
parser.add_argument('--output_csv', required=True)
parser.add_argument('--lang', help='language of file to predict', required=True)
parser.add_argument('--only_high_logits', action='store_true', help='only high logits outputs were kept')
args = parser.parse_args()

predicted_files =[f'../stance_expts/ml-stance-11_en-no_mlm-2_neg-rs_rp-variant_3-attempt_r3/checkpoint-best/test-{args.lang}.tsv',
                  f'../stance_expts/ml-stance-11_en-no_mlm-2_neg-rs_rp-ld_mldoc-extra_mlm-amazonzh-variant_3-attempt_r4/checkpoint-best/test-{args.lang}.tsv',
                  f'../stance_expts/ml-stance-11_en-no_mlm-2_neg-rs_rp-variant_3-large_model-attempt_r3/checkpoint-best/test-{args.lang}.tsv']

pred_dfs = [pd.read_csv(i, names=['pred', 'text'], index_col=False, sep='\t') for i in predicted_files]
orig_df = pd.read_csv(args.orig_file, lineterminator='\n', index_col=0)

n_rows = len(orig_df)
majority = []
text = []
topic = []

if __debug__ or args.only_high_logits:
    import transformers
    tokenizer = transformers.XLMRobertaTokenizer.from_pretrained('microsoft/infoxlm-base')

if args.only_high_logits:
    pred_indexes = [0, 0, 0]
    for idx in tqdm.trange(n_rows):
        label_list = []
        try:
            orig_text = orig_df.iloc[idx]['text'].strip()
        except AttributeError:
            continue
        orig_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokenizer(orig_df.iloc[idx]['text'], add_special_tokens=False)['input_ids']))
        work_text = orig_df.iloc[idx]['text'].strip()
        text.append(work_text)
        topic.append(orig_df.iloc[idx]['search_term'])
        for df_i, df in enumerate(pred_dfs):
            try:
                p_text = df.iloc[pred_indexes[df_i]]['text'].split('Stance of ')[1].split('is<mask>')[0].strip()
            except IndexError:
                continue
            if orig_text.startswith(p_text):
                label_list.append(df.iloc[pred_indexes[df_i]]['pred'])
                pred_indexes[df_i] += 1
        if len(label_list) == 2:
            if label_list[0] == label_list[1]:
                majority.append(label_list[0])
            else:
                majority.append('drop')
        elif len(label_list) == 3:
            if label_list[0] == label_list[1] or label_list[0] == label_list[2]:
                majority.append(label_list[0])
            elif label_list[1] == label_list[2]:
                majority.append(label_list[1])
            else:
                majority.append('drop')
        else:
            majority.append('drop')

else:
    shift = 0
    for idx in tqdm.trange(n_rows):
        try:
            assert len(set([df.iloc[idx-shift]['text'] for df in pred_dfs])) == 1
        except AssertionError:
            continue
        if __debug__:
            pred_text = pred_dfs[0].iloc[idx-shift]['text'].split('Stance of ')[1].split('is<mask>')[0].strip()
            try:
                orig_text = orig_df.iloc[idx]['text'].strip()
            except AttributeError as e:
                shift += 1
                print(idx)
                continue
            if pred_text == orig_text:
                pass
            else:
                orig_text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(tokenizer(orig_df.iloc[idx]['text'], add_special_tokens=False)['input_ids']))
                if not orig_text.startswith(pred_text):
                    raise AssertionError((pred_text, orig_text))
        prediction = [df.iloc[idx-shift]['pred'] for df in pred_dfs]
        work_text = orig_df.iloc[idx]['text'].strip()
        text.append(work_text)
        topic.append(orig_df.iloc[idx]['search_term'])
        if prediction[0] == prediction[1] or prediction[0] == prediction[2]:
            majority.append(prediction[0])
        elif prediction[1] == prediction[2]:
            majority.append(prediction[1])
        else:
            majority.append('drop')

final_df = pd.DataFrame(list(zip(majority, text, topic)), columns=['pred', 'text', 'topic'])
final_df = final_df[final_df['pred'] != 'drop']
final_df.to_csv(args.output_csv, index=False)
print(final_df.shape)

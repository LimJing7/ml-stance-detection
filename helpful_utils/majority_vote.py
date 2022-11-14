import pandas as pd
predicted_files =['../stance_expts/ml-stance-11_en-no_mlm-2_neg-rs_rp-variant_3-attempt_r3/checkpoint-best/test-zh.tsv',
                  '../stance_expts/ml-stance-11_en-no_mlm-2_neg-rs_rp-ld_mldoc-extra_mlm-amazonzh-variant_3-attempt_r4/checkpoint-best/test-zh.tsv',
                  '../stance_expts/ml-stance-11_en-no_mlm-2_neg-rs_rp-variant_3-large_model-attempt_r3/checkpoint-best/test-zh.tsv']

pred_dfs = [pd.read_csv(i, names=['pred', 'text'], index_col=False, sep='\t') for i in predicted_files]

n_rows = len(pred_dfs[0])
majority = []
text = []

for idx in range(n_rows):
    assert(len(set([df.iloc[idx]['text'] for df in pred_dfs])) == 1)
    prediction = [df.iloc[idx]['pred'] for df in pred_dfs]
    work_text = pred_dfs[0].iloc[idx]['text']
    text.append(work_text[14:-26])
    if prediction[0] == prediction[1] or prediction[0] == prediction[2]:
        majority.append(prediction[0])
    elif prediction[1] == prediction[2]:
        majority.append(prediction[1])
    else:
        majority.append('drop')

final_df = pred_dfs[0]
final_df['pred'] = majority
final_df['text'] = text
final_df = final_df[final_df['pred'] != 'drop']
final_df.to_csv('majority_voted_twitter_iphonese_zh.csv')
print(final_df.shape)
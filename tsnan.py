import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TSNAN(nn.Module):
    '''Model is adapted Stance Classification with Target-Specific Neural Attention Networks
    '''
    def __init__(self, n_emb, embed_size, lstm_size, output_size, dropout_rate) -> None:
        super().__init__()

        self.embedding = torch.nn.Embedding(n_emb, embed_size)
        self.lstm = torch.nn.LSTM(input_size=embed_size, hidden_size=lstm_size, bidirectional=True, batch_first=True)

        self.target_linear = torch.nn.Linear(2*embed_size, 1)
        self.target_softmax = torch.nn.Softmax(dim=1)

        self.output_linear = torch.nn.Linear(2*lstm_size, output_size)

        self.dropout = torch.nn.Dropout(p=dropout_rate)

        # self.apply(self._init_weights)

    def forward(self, topic, topic_mask, text, text_mask):
        topic_lens = torch.sum(topic_mask, axis=1, keepdim=True)  # batch * 1
        text_lens = torch.sum(text_mask, axis=1, keepdim=True)  # batch * 1

        text_emb = self.embedding(text)  # batch * len * emb
        text_emb_packed = pack_padded_sequence(text_emb, text_lens.squeeze().cpu(), batch_first=True, enforce_sorted=False)

        topic_emb = self.embedding(topic)  # batch * len * emb
        topic_emb = torch.mul(topic_emb, topic_mask.unsqueeze(2))
        topic_emb = torch.sum(topic_emb, axis=1, keepdim=True)  # batch * 1 * emb
        topic_emb = torch.div(topic_emb, topic_lens.unsqueeze(2)).expand(text_emb.shape)  # batch * 1 * emb

        lstm_out, states = self.lstm(text_emb_packed)
        lstm_out = pad_packed_sequence(lstm_out, batch_first=True, total_length=text_mask.shape[1])[0]
        lstm_out = self.dropout(lstm_out)

        cat_emb = torch.cat([text_emb, topic_emb], dim=2)
        attn = self.target_linear(cat_emb)  # batch * len * 1
        mask = torch.zeros_like(text_mask, dtype=torch.float32)
        mask[~text_mask.bool()] = float('-inf')
        attn = attn + mask.unsqueeze(2)
        # attn = torch.mul(attn, text_mask.unsqueeze(2))  # mask out padding
        attn = self.target_softmax(attn)

        comb = torch.mul(lstm_out, attn)
        sent_emb = torch.sum(comb, dim=1)  # batch * 2lstm_size
        sent_emb = torch.div(sent_emb, text_lens)
        sent_emb = self.dropout(sent_emb)

        output = self.output_linear(sent_emb)

        return output

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Embedding):
            torch.nn.init.uniform_(module.weight, -0.01, 0.01)
        elif isinstance(module, torch.nn.LSTM):
            for weight in module.parameters():
                torch.nn.init.uniform_(weight, -0.01, 0.01)
        elif isinstance(module, torch.nn.Linear):
            torch.nn.init.uniform_(module.weight, -0.01, 0.01)
            if module.bias is not None:
                torch.nn.init.uniform_(module.weight, -0.01, 0.01)


class HalfTSNAN(nn.Module):
    '''Model is half of Stance Classification with Target-Specific Neural Attention Networks
    with the lstm and embedding layers removed
    '''
    def __init__(self, embed_size, hidden_size, output_size) -> None:
        super().__init__()
        self.target_linear = torch.nn.Linear(2*embed_size, 1)
        self.target_softmax = torch.nn.Softmax(dim=1)

        self.output_linear = torch.nn.Linear(hidden_size, output_size)



    def forward(self, topic_emb, topic_mask, text_emb, text_mask):
        topic_lens = torch.sum(topic_mask, axis=1, keepdim=True)  # batch * 1
        text_lens = torch.sum(text_mask, axis=1, keepdim=True)  # batch * 1]

        topic_emb = torch.mul(topic_emb, topic_mask.unsqueeze(2))
        topic_emb = torch.sum(topic_emb, axis=1, keepdim=True)  # batch * 1 * emb
        topic_emb = torch.div(topic_emb, topic_lens.unsqueeze(2)).expand(text_emb.shape)  # batch * 1 * emb


        cat_emb = torch.cat([text_emb, topic_emb], dim=2)
        attn = self.target_linear(cat_emb)  # batch * len * 1
        mask = torch.zeros_like(text_mask, dtype=torch.float32)
        mask[~text_mask.bool()] = float('-inf')
        attn = attn + mask.unsqueeze(2)
        attn = self.target_softmax(attn)

        comb = torch.mul(text_emb, attn)
        output = self.output_linear(comb)

        return output
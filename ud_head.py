## adapted from github.com/yzhangcs/parser

from collections import OrderedDict
import copy
from typing import Optional, Any

import torch
from torch import nn
from torch import Tensor
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.modules.rnn import apply_permutation
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence
import transformers
from transformers import  CamembertConfig, XLMRobertaConfig, BertConfig, RobertaConfig, XLMConfig
from transformers import BertModel, RobertaModel, BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead, BertPooler
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel

MLP_ARC_SIZE = 500
MLP_LABEL_SIZE = 100
MLP_DROPOUT = 0.33

UD_LABEL_COUNT = 37
UD_LABEL_DICT = {
    'acl': 0,
    'acl:relcl': 0,
    'advcl': 1,
    'advmod': 2,
    'amod': 3,
    'appos': 4,
    'aux': 5,
    'aux:pass': 5,
    'case': 6,
    'cc': 7,
    'ccomp': 8,
    'clf': 9,
    'compound': 10,
    'compound:prt': 10,
    'conj': 11,
    'cop': 12,
    'csubj': 13,
    'csubj:pass': 13,
    'dep': 14,
    'det': 15,
    'det:predet': 15,
    'discourse': 16,
    'dislocated': 17,
    'expl': 18,
    'fixed': 19,
    'flat': 20,
    'goeswith': 21,
    'iobj': 22,
    'list': 23,
    'mark': 24,
    'nmod': 25,
    'nmod:poss': 25,
    'nmod:tmod': 25,
    'nsubj': 26,
    'nsubj:pass': 26,
    'nummod': 27,
    'obj': 28,
    'obl': 29,
    'obl:npmod': 29,
    'orphan': 30,
    'parataxis': 31,
    'punct': 32,
    'reparandum': 33,
    'root': 34,
    'vocative': 35,
    'xcomp': 36,
}
UD_LABEL_LIST = [
    'acl',
    'advcl',
    'advmod',
    'amod',
    'appos',
    'aux',
    'case',
    'cc',
    'ccomp',
    'clf',
    'compound',
    'conj',
    'cop',
    'csubj',
    'dep',
    'det',
    'discourse',
    'dislocated',
    'expl',
    'fixed',
    'flat',
    'goeswith',
    'iobj',
    'list',
    'mark',
    'nmod',
    'nsubj',
    'nummod',
    'obj',
    'obl',
    'orphan',
    'parataxis',
    'punct',
    'reparandum',
    'root',
    'vocative',
    'xcomp'
]


class FullUDHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        try:
            self.n_layers = min(5, config.first_layers)
        except AttributeError:
            self.n_layers = 5
        self.scalar_mix = ScalarMix(self.n_layers)
        try:
            ud_layers = config.ud_layers
        except AttributeError:
            ud_layers = 3
        self.lstm = BiLSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=3)

        self.mlp_arc_head = MLP(config.hidden_size*2, MLP_ARC_SIZE, MLP_DROPOUT)
        self.mlp_arc_dep = MLP(config.hidden_size*2, MLP_ARC_SIZE, MLP_DROPOUT)
        self.mlp_rel_head = MLP(config.hidden_size*2, MLP_LABEL_SIZE, MLP_DROPOUT)
        self.mlp_rel_dep = MLP(config.hidden_size*2, MLP_LABEL_SIZE, MLP_DROPOUT)

        self.arc_attn = Biaffine(MLP_ARC_SIZE, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(MLP_LABEL_SIZE, n_out=UD_LABEL_COUNT, bias_x=True, bias_y=True)

    def forward(self, hidden_states, lens):
        ## hidden_states shape = tuple(b*l*hidden_size)*nLayers

        selected_layers = hidden_states[-self.n_layers:]
        device = hidden_states[0].device

        states_mix = self.scalar_mix(selected_layers)
        b, l, h = states_mix.shape

        # lf = lens.flatten()
        # bert_output = torch.zeros(b, l, max(lf), h).to(device)
        words_output = torch.zeros(b, l, h)
        mask = torch.zeros(b).long()

        for i in range(b):
            counter = 0
            item = []
            for word_len in lens[i]:
                if word_len > 0:
                    bl = states_mix[i, counter:counter+word_len]
                    item.append(torch.mean(bl, axis=0))
                    counter += word_len
            mask[i] = len(item)
            words_output[i, :len(item)] = torch.stack(item)

        mask = mask.to(device)
        words_output = words_output.to(device)

        x = pack_padded_sequence(words_output, mask.cpu(), batch_first=True, enforce_sorted=False)
        x, hx = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=l)

        arc_head = self.mlp_arc_head(x)
        arc_dep = self.mlp_arc_dep(x)
        rel_head = self.mlp_rel_head(x)
        rel_dep = self.mlp_rel_dep(x)

        s_arc = self.arc_attn(arc_dep, arc_head) # b * len * len
        s_rel = self.rel_attn(rel_dep, rel_head).permute(0, 2, 3, 1) # b * len * len * label

        return s_arc, s_rel


class BertForFullUD(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.ud = FullUDHead(config)

        try:
            self.first_layers = config.first_layers
        except AttributeError:
            pass

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        ud_arc=None,
        ud_rel=None,
        tok_lens=None,
    ):
        r"""
        ud_arc (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the head of the word.
            Indices should be in ``[-100, 0, ..., sequence_length]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., sequence_length]``
        ud_rel (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing dependency type.
            Indices should be in ``[-100, 0, ..., UD_LABEL_COUNT]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., UD_LABEL_COUNT]``
        tok_lens (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Number of tokens for each word
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )

        sequence_output = outputs[0]
        if tok_lens is not None:
            if self.first_layers:
                hidden_states = outputs[2][:self.first_layers]
            else:
                hidden_states = outputs[2]
            s_arc, s_rel = self.ud(hidden_states, tok_lens)
            outputs = ((s_arc, s_rel),) + outputs[2:]

            if ud_arc is not None and ud_rel is not None:
                loss_fct = CrossEntropyLoss()
                arc_loss = loss_fct(s_arc.permute(0,2,1), ud_arc)
                # s_rel = s_rel[torch.arange(len(ud_arc)), ud_arc]
                gather_idx = ud_arc.clone()
                gather_idx[ud_arc.eq(-100)] = 0
                gather_idx = torch.stack([gather_idx.unsqueeze(-1)]*UD_LABEL_COUNT, 3)
                s_rel = s_rel.gather(2, gather_idx).squeeze(2)
                rel_loss = loss_fct(s_rel.permute(0,2,1), ud_rel)
                total_loss = arc_loss + rel_loss
                outputs = (total_loss,) + outputs

        return outputs  # (loss), (s_arc, s_rel), (hidden_states), (attentions)

class BertForFullUDwTrans(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        self.trans = TransformerEncoder(enc_layer, num_layers=4) # rmb to permute
        self.ud = FullUDHead(config)

        self.first_layers = config.first_layers

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
        ud_arc=None,
        ud_rel=None,
        tok_lens=None,
    ):
        r"""
        ud_arc (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the head of the word.
            Indices should be in ``[-100, 0, ..., sequence_length]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., sequence_length]``
        ud_rel (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing dependency type.
            Indices should be in ``[-100, 0, ..., UD_LABEL_COUNT]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., UD_LABEL_COUNT]``
        tok_lens (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Number of tokens for each word
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )

        sequence_output = outputs[0]
        if tok_lens is not None:
            hidden_states = outputs[2]
            pytorch_enc_mask = (1-attention_mask).bool()
            outs = self.trans(hidden_states[self.first_layers].permute(1, 0, 2),
                              src_key_padding_mask=pytorch_enc_mask)
            outs = list(map(lambda x: x.permute(1, 0, 2), outs))
            s_arc, s_rel = self.ud(outs, tok_lens)
            outputs = ((s_arc, s_rel),) + outputs[2:]

            if ud_arc is not None and ud_rel is not None:
                loss_fct = CrossEntropyLoss()
                arc_loss = loss_fct(s_arc.permute(0,2,1), ud_arc)
                # s_rel = s_rel[torch.arange(len(ud_arc)), ud_arc]
                gather_idx = ud_arc.clone()
                gather_idx[ud_arc.eq(-100)] = 0
                gather_idx = torch.stack([gather_idx.unsqueeze(-1)]*UD_LABEL_COUNT, 3)
                s_rel = s_rel.gather(2, gather_idx).squeeze(2)
                rel_loss = loss_fct(s_rel.permute(0,2,1), ud_rel)
                total_loss = arc_loss + rel_loss
                outputs = (total_loss,) + outputs

        return outputs  # (loss), (s_arc, s_rel), (hidden_states), (attentions)

class BertForFullUDwMapTrans(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.map = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        self.trans = TransformerEncoder(enc_layer, num_layers=4) # rmb to permute
        self.ud = FullUDHead(config)

        self.first_layers = config.first_layers

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
        ud_arc=None,
        ud_rel=None,
        tok_lens=None,
    ):
        r"""
        ud_arc (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the head of the word.
            Indices should be in ``[-100, 0, ..., sequence_length]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., sequence_length]``
        ud_rel (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing dependency type.
            Indices should be in ``[-100, 0, ..., UD_LABEL_COUNT]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., UD_LABEL_COUNT]``
        tok_lens (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Number of tokens for each word
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )

        sequence_output = outputs[0]
        if tok_lens is not None:
            hidden_states = outputs[2]
            pytorch_enc_mask = (1-attention_mask).bool()
            # print('unmapped: \n')
            # print(hidden_states[self.first_layers])
            mapped = self.map(hidden_states[self.first_layers])
            # print('mapped:\n')
            # print(mapped)
            outs = self.trans(mapped.permute(1, 0, 2),
                              src_key_padding_mask=pytorch_enc_mask)
            outs = list(map(lambda x: x.permute(1, 0, 2), outs))
            s_arc, s_rel = self.ud(outs, tok_lens)
            outputs = ((s_arc, s_rel),) + outputs[2:]

            if ud_arc is not None and ud_rel is not None:
                loss_fct = CrossEntropyLoss()
                arc_loss = loss_fct(s_arc.permute(0,2,1), ud_arc)
                # s_rel = s_rel[torch.arange(len(ud_arc)), ud_arc]
                gather_idx = ud_arc.clone()
                gather_idx[ud_arc.eq(-100)] = 0
                gather_idx = torch.stack([gather_idx.unsqueeze(-1)]*UD_LABEL_COUNT, 3)
                s_rel = s_rel.gather(2, gather_idx).squeeze(2)
                rel_loss = loss_fct(s_rel.permute(0,2,1), ud_rel)
                total_loss = arc_loss + rel_loss
                outputs = (total_loss,) + outputs

        return outputs  # (loss), (s_arc, s_rel), (hidden_states), (attentions)

class CamembertForFullUD(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.ud = FullUDHead(config)

        self.first_layers = config.first_layers

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        ud_arc=None,
        ud_rel=None,
        tok_lens=None,
    ):
        r"""
        ud_arc (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the head of the word.
            Indices should be in ``[-100, 0, ..., sequence_length]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., sequence_length]``
        ud_rel (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing dependency type.
            Indices should be in ``[-100, 0, ..., UD_LABEL_COUNT]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., UD_LABEL_COUNT]``
        tok_lens (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Number of tokens for each word
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )

        sequence_output = outputs[0]
        if tok_lens is not None:
            if self.first_layers:
                hidden_states = outputs[2][:self.first_layers]
            else:
                hidden_states = outputs[2]
            s_arc, s_rel = self.ud(hidden_states, tok_lens)
            outputs = ((s_arc, s_rel),) + outputs[2:]

            if ud_arc is not None and ud_rel is not None:
                loss_fct = CrossEntropyLoss()
                arc_loss = loss_fct(s_arc.permute(0,2,1), ud_arc)
                # s_rel = s_rel[torch.arange(len(ud_arc)), ud_arc]
                gather_idx = ud_arc.clone()
                gather_idx[ud_arc.eq(-100)] = 0
                gather_idx = torch.stack([gather_idx.unsqueeze(-1)]*UD_LABEL_COUNT, 3)
                s_rel = s_rel.gather(2, gather_idx).squeeze(2)
                rel_loss = loss_fct(s_rel.permute(0,2,1), ud_rel)
                total_loss = arc_loss + rel_loss
                outputs = (total_loss,) + outputs

        return outputs  # (loss), (s_arc, s_rel), (hidden_states), (attentions)


class CamembertForFullUDwMapTrans(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.map = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        self.trans = TransformerEncoder(enc_layer, num_layers=4) # rmb to permute
        self.ud = FullUDHead(config)

        self.first_layers = config.first_layers

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
        ud_arc=None,
        ud_rel=None,
        tok_lens=None,
    ):
        r"""
        ud_arc (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the head of the word.
            Indices should be in ``[-100, 0, ..., sequence_length]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., sequence_length]``
        ud_rel (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing dependency type.
            Indices should be in ``[-100, 0, ..., UD_LABEL_COUNT]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., UD_LABEL_COUNT]``
        tok_lens (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Number of tokens for each word
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )

        sequence_output = outputs[0]
        if tok_lens is not None:
            hidden_states = outputs[2]
            pytorch_enc_mask = (1-attention_mask).bool()
            # print('unmapped: \n')
            # print(hidden_states[self.first_layers])
            mapped = self.map(hidden_states[self.first_layers])
            # print('mapped:\n')
            # print(mapped)
            outs = self.trans(mapped.permute(1, 0, 2),
                              src_key_padding_mask=pytorch_enc_mask)
            outs = list(map(lambda x: x.permute(1, 0, 2), outs))
            s_arc, s_rel = self.ud(outs, tok_lens)
            outputs = ((s_arc, s_rel),) + outputs[2:]

            if ud_arc is not None and ud_rel is not None:
                loss_fct = CrossEntropyLoss()
                arc_loss = loss_fct(s_arc.permute(0,2,1), ud_arc)
                # s_rel = s_rel[torch.arange(len(ud_arc)), ud_arc]
                gather_idx = ud_arc.clone()
                gather_idx[ud_arc.eq(-100)] = 0
                gather_idx = torch.stack([gather_idx.unsqueeze(-1)]*UD_LABEL_COUNT, 3)
                s_rel = s_rel.gather(2, gather_idx).squeeze(2)
                rel_loss = loss_fct(s_rel.permute(0,2,1), ud_rel)
                total_loss = arc_loss + rel_loss
                outputs = (total_loss,) + outputs

        return outputs  # (loss), (s_arc, s_rel), (hidden_states), (attentions)


class RobertaForFullUD(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.ud = FullUDHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        ud_arc=None,
        ud_rel=None,
        tok_lens=None,
    ):
        r"""
        ud_arc (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the head of the word.
            Indices should be in ``[-100, 0, ..., sequence_length]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., sequence_length]``
        ud_rel (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing dependency type.
            Indices should be in ``[-100, 0, ..., UD_LABEL_COUNT]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., UD_LABEL_COUNT]``
        tok_lens (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Number of tokens for each word
        """

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )

        sequence_output = outputs[0]
        if tok_lens is not None:
            hidden_states = outputs[2]
            s_arc, s_rel = self.ud(hidden_states, tok_lens)
            outputs = ((s_arc, s_rel),) + outputs[2:]

            if ud_arc is not None and ud_rel is not None:
                loss_fct = CrossEntropyLoss()
                arc_loss = loss_fct(s_arc.permute(0,2,1), ud_arc)
                # s_rel = s_rel[torch.arange(len(ud_arc)), ud_arc]
                gather_idx = ud_arc.clone()
                gather_idx[ud_arc.eq(-100)] = 0
                gather_idx = torch.stack([gather_idx.unsqueeze(-1)]*UD_LABEL_COUNT, 3)
                s_rel = s_rel.gather(2, gather_idx).squeeze(2)
                rel_loss = loss_fct(s_rel.permute(0,2,1), ud_rel)
                total_loss = arc_loss + rel_loss
                outputs = (total_loss,) + outputs

        return outputs  # (loss), (s_arc, s_rel), (hidden_states), (attentions)

    def get_lang_dep(self):
        try:
            return self.other
        except AttributeError:
            return None

    def set_lang_dep(self, model_module):
        # not used just keep
        self.other = model_module

    def set_main(self, model_module):
        pass


class XLMRobertaForFullUD(RobertaForFullUD):

    config_class = XLMRobertaConfig



class BertForFullUDAndMLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.ud = FullUDHead(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
        ud_arc=None,
        ud_rel=None,
        tok_lens=None,
    ):
        r"""
        ud_arc (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the head of the word.
            Indices should be in ``[-100, 0, ..., sequence_length]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., sequence_length]``
        ud_rel (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing dependency type.
            Indices should be in ``[-100, 0, ..., UD_LABEL_COUNT]``
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., UD_LABEL_COUNT]``
        tok_lens (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Number of tokens for each word

        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_hidden_states=True,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        if tok_lens is not None:
            hidden_states = outputs[2]
            s_arc, s_rel = self.ud(hidden_states, tok_lens)
            outputs = ((prediction_scores, s_arc, s_rel), ) + outputs[2:]
        else:
            outputs = (prediction_scores,) + outputs[2:]
        # batch*?, batch * len * len, batch * len * len * label

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs


        if tok_lens is not None and ud_arc is not None and ud_rel is not None:
            loss_fct = CrossEntropyLoss()
            arc_loss = loss_fct(s_arc.permute(0,2,1), ud_arc)
            # s_rel = s_rel[torch.arange(len(ud_arc)), ud_arc]
            gather_idx = ud_arc.clone()
            gather_idx[ud_arc.eq(-100)] = 0
            gather_idx = torch.stack([gather_idx.unsqueeze(-1)]*UD_LABEL_COUNT, 3)
            s_rel = s_rel.gather(2, gather_idx).squeeze(2).permute(0,2,1)
            rel_loss = loss_fct(s_rel, ud_rel)
            total_loss = arc_loss + rel_loss
            outputs = (total_loss,) + outputs

        return outputs  # (ud_loss), (masked_lm_loss), (ltr_lm_loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)

class BertForFullUDAndSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.ud = FullUDHead(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        ud_arc=None,
        ud_rel=None,
        tok_lens=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        pooled_output = outputs[1]
        hidden_states = outputs[2]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs


        if tok_lens is not None:
            s_arc, s_rel = self.ud(hidden_states, tok_lens)
            outputs = ((s_arc, s_rel),) + outputs

            if ud_arc is not None and ud_rel is not None:
                loss_fct = CrossEntropyLoss()
                arc_loss = loss_fct(s_arc.permute(0,2,1), ud_arc)
                # s_rel = s_rel[torch.arange(len(ud_arc)), ud_arc]
                gather_idx = ud_arc.clone()
                gather_idx[ud_arc.eq(-100)] = 0
                gather_idx = torch.stack([gather_idx.unsqueeze(-1)]*UD_LABEL_COUNT, 3)
                s_rel = s_rel.gather(2, gather_idx).squeeze(2).permute(0,2,1)
                rel_loss = loss_fct(s_rel, ud_rel)
                total_loss = arc_loss + rel_loss
                outputs = (total_loss,) + outputs

        return outputs  # (loss), (s_arc, s_rel), logits, (hidden_states), (attentions)


class BertForSequenceClassificationwMapTrans(BertPreTrainedModel):  ## check this before use
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.map = torch.nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        self.trans = TransformerEncoder(enc_layer, num_layers=4) # rmb to permute
        self.pooler = BertPooler(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
        )

        hidden_states = outputs[2]
        pytorch_enc_mask = (1-attention_mask).bool()
        mapped = self.map(hidden_states[self.first_layers])
        outs = self.trans(mapped.permute(1, 0, 2),
                            src_key_padding_mask=pytorch_enc_mask)
        outs = list(map(lambda x: x.permute(1, 0, 2), outs))

        pooled_output = self.pooler(outs[-1])

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class Biaffine(nn.Module):

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(Biaffine, self).__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out,
                                                n_in + bias_x,
                                                n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s

class BiLSTM(nn.Module):
    """
    The BiLSTM module is an variant of the vanilla bidirectional LSTM adopted by Biaffine Parser.
    The difference between them is the dropout strategy.
    This module drops nodes in the LSTM layers (input and recurrent connections)
    and applies the same dropout mask at every recurrent timesteps.
    APIs are roughly the same as nn.LSTM except that we remove the `bidirectional` option
    and only allows PackedSequence as input.
    References:
        - Timothy Dozat and Christopher D. Manning (ICLR'17)
          Deep Biaffine Attention for Neural Dependency Parsing
          https://openreview.net/pdf?id=Hk95PK9le/
    Args:
        input_size (int):
            The number of expected features in the input.
        hidden_size (int):
            The number of features in the hidden state h.
        num_layers (int):
            Number of recurrent layers. Default: 1.
        dropout (float):
            If non-zero, introduces a SharedDropout layer on the outputs of each LSTM layer
            except the last layer, with dropout probability equal to `dropout`.
            Default: 0.
    """

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        self.f_cells = nn.ModuleList()
        self.b_cells = nn.ModuleList()
        for _ in range(self.num_layers):
            self.f_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            self.b_cells.append(nn.LSTMCell(input_size=input_size,
                                            hidden_size=hidden_size))
            input_size = hidden_size * 2

        self.reset_parameters()

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f"{self.input_size}, {self.hidden_size}"
        if self.num_layers > 1:
            s += f", num_layers={self.num_layers}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"
        s += ')'

        return s

    def reset_parameters(self):
        for param in self.parameters():
            # apply orthogonal_ to weight
            if len(param.shape) > 1:
                nn.init.orthogonal_(param)
            # apply zeros_ to bias
            else:
                nn.init.zeros_(param)

    def permute_hidden(self, hx, permutation):
        if permutation is None:
            return hx
        h = apply_permutation(hx[0], permutation)
        c = apply_permutation(hx[1], permutation)

        return h, c

    def layer_forward(self, x, hx, cell, batch_sizes, reverse=False):
        hx_0 = hx_i = hx
        hx_n, output = [], []
        steps = reversed(range(len(x))) if reverse else range(len(x))
        if self.training:
            hid_mask = SharedDropout.get_mask(hx_0[0], self.dropout)

        for t in steps:
            last_batch_size, batch_size = len(hx_i[0]), batch_sizes[t]
            if last_batch_size < batch_size:
                hx_i = [torch.cat((h, ih[last_batch_size:batch_size]))
                        for h, ih in zip(hx_i, hx_0)]
            else:
                hx_n.append([h[batch_size:] for h in hx_i])
                hx_i = [h[:batch_size] for h in hx_i]
            hx_i = [h for h in cell(x[t], hx_i)]
            output.append(hx_i[0])
            if self.training:
                hx_i[0] = hx_i[0] * hid_mask[:batch_size]
        if reverse:
            hx_n = hx_i
            output.reverse()
        else:
            hx_n.append(hx_i)
            hx_n = [torch.cat(h) for h in zip(*reversed(hx_n))]
        output = torch.cat(output)

        return output, hx_n

    def forward(self, sequence, hx=None):
        """
        Args:
            sequence (PackedSequence):
                A packed variable length sequence.
            hx=(h, x) (Tuple[Tensor, Tensor]):
                h ([num_layers * 2, batch_size, hidden_size]) contains the initial hidden state
                for each element in the batch.
                c ([num_layers * 2, batch_size, hidden_size]) contains the initial cell state
                for each element in the batch.
                If (h, x) is not provided, both h and c default to zero.
                Default: None.
        Returns:
            x (PackedSequence):
                A packed variable length sequence.
            hx=(h, x) (Tuple[Tensor, Tensor]):
                h ([num_layers * 2, batch_size, hidden_size]) contains the hidden state for `t = seq_len`.
                Like output, the layers can be separated using h.view(num_layers, 2, batch_size, hidden_size)
                and similarly for c.
                c ([num_layers * 2, batch_size, hidden_size]) contains the cell state for `t = seq_len`.
        """
        x, batch_sizes = sequence.data, sequence.batch_sizes.tolist()
        batch_size = batch_sizes[0]
        h_n, c_n = [], []

        if hx is None:
            ih = x.new_zeros(self.num_layers * 2, batch_size, self.hidden_size)
            h, c = ih, ih
        else:
            h, c = self.permute_hidden(hx, sequence.sorted_indices)
        h = h.view(self.num_layers, 2, batch_size, self.hidden_size)
        c = c.view(self.num_layers, 2, batch_size, self.hidden_size)

        for i in range(self.num_layers):
            x = torch.split(x, batch_sizes)
            if self.training:
                mask = SharedDropout.get_mask(x[0], self.dropout)
                x = [i * mask[:len(i)] for i in x]
            x_f, (h_f, c_f) = self.layer_forward(x=x,
                                                 hx=(h[i, 0], c[i, 0]),
                                                 cell=self.f_cells[i],
                                                 batch_sizes=batch_sizes)
            x_b, (h_b, c_b) = self.layer_forward(x=x,
                                                 hx=(h[i, 1], c[i, 1]),
                                                 cell=self.b_cells[i],
                                                 batch_sizes=batch_sizes,
                                                 reverse=True)
            x = torch.cat((x_f, x_b), -1)
            h_n.append(torch.stack((h_f, h_b)))
            c_n.append(torch.stack((c_f, c_b)))
        x = PackedSequence(x,
                           sequence.batch_sizes,
                           sequence.sorted_indices,
                           sequence.unsorted_indices)
        hx = torch.cat(h_n, 0), torch.cat(c_n, 0)
        hx = self.permute_hidden(hx, sequence.unsorted_indices)

        return x, hx

class MLP(nn.Module):

    def __init__(self, n_in, n_hidden, dropout=0):
        super(MLP, self).__init__()

        self.linear = nn.Linear(n_in, n_hidden)
        self.activation = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = SharedDropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x

class SharedDropout(nn.Module):

    def __init__(self, p=0.5, batch_first=True):
        super(SharedDropout, self).__init__()

        self.p = p
        self.batch_first = batch_first

    def extra_repr(self):
        s = f"p={self.p}"
        if self.batch_first:
            s += f", batch_first={self.batch_first}"

        return s

    def forward(self, x):
        if self.training:
            if self.batch_first:
                mask = self.get_mask(x[:, 0], self.p)
            else:
                mask = self.get_mask(x[0], self.p)
            x *= mask.unsqueeze(1) if self.batch_first else mask

        return x

    @staticmethod
    def get_mask(x, p):
        mask = x.new_empty(x.shape).bernoulli_(1 - p)
        mask = mask / (1 - p)

        return mask

class ScalarMix(nn.Module):
    """
    Compute a parameterised scalar mixture of N tensors, `mixture = gamma * sum(s_k * tensor_k)`
    where `s = softmax(w)`, with `w` and `gamma` scalar parameters.
    Args:
        n_layers (int):
            Number of layers to be mixed, i.e., N.
        dropout (float):
            The dropout ratio of the layer weights.
            If dropout > 0, then for each scalar weight, adjust its softmax weight mass to 0
            with the dropout probability (i.e., setting the unnormalized weight to -inf).
            This effectively redistributes the dropped probability mass to all other weights.
            Default: 0.
    """

    def __init__(self, n_layers, dropout=0):
        super().__init__()

        self.n_layers = n_layers

        self.weights = nn.Parameter(torch.zeros(n_layers))
        self.scale = nn.Parameter(torch.tensor([1.0])) # gamma has issues
        self.dropout = nn.Dropout(dropout)

    def extra_repr(self):
        s = f"n_layers={self.n_layers}"
        if self.dropout.p > 0:
            s += f", dropout={self.dropout.p}"

        return s

    def forward(self, tensors):
        """
        Args:
            tensors (List[Tensor]):
                N tensors to be mixed.
        Returns:
            The mixture of N tensors.
        """

        normed_weights = self.dropout(self.weights.softmax(-1))
        weighted_sum = sum(w * h for w, h in zip(normed_weights, tensors))

        return self.scale * weighted_sum


class TransformerEncoder(nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers

    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        outputs  = [src]
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            outputs.append(output)

        if self.norm is not None:
            output = self.norm(output)  # not used atm

        return outputs


def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

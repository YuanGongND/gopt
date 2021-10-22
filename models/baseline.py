# -*- coding: utf-8 -*-
# @Time    : 9/28/21 2:14 PM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : baseline.py

# baseline models

import torch
import torch.nn as nn

class BaselineNN(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, input_dim=84):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim)) for i in range(depth)])

        # for phone classification
        self.in_proj = nn.Linear(self.input_dim, embed_dim)
        self.mlp_head_phn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # for word classification
        self.mlp_head_word1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # phone projection
        self.phn_proj = nn.Linear(40, embed_dim)

        # utterance level
        self.mlp_head_utt1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt4 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt5 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

    # get the avg of only the valid token
    def apply_mean_mask(self, input, mask):
        input = input * mask
        input = torch.sum(input, dim=1)
        input = input / torch.sum(mask, dim=1)
        return input.unsqueeze(1)

    # x shape in [batch_size, sequence_len, feat_dim]
    # phn in [batch_size, seq_len]
    def forward(self, x, phn):

        # batch size
        B = x.shape[0]
        seq_len = x.shape[1]
        valid_tok_mask = (phn>=0)
        #print(valid_tok_mask.shape)

        # phn_one_hot in shape [batch_size, seq_len, feat_dim]
        phn_one_hot =  torch.nn.functional.one_hot(phn.long()+1, num_classes=40).float()
        # phn_embed in shape [batch_size, seq_len, embed_dim]
        phn_embed = self.phn_proj(phn_one_hot)

        if self.embed_dim != self.input_dim:
            x = self.in_proj(x)

        x = x + phn_embed

        x = x.reshape(B * seq_len, self.embed_dim)

        for blk in self.blocks:
            x = blk(x)

        p = self.mlp_head_phn(x).reshape(B, seq_len, 1)

        w1 = self.mlp_head_word1(x).reshape(B, seq_len, 1)
        w2 = self.mlp_head_word2(x).reshape(B, seq_len, 1)
        w3 = self.mlp_head_word3(x).reshape(B, seq_len, 1)

        u1 = self.apply_mean_mask(self.mlp_head_utt1(x).reshape(B, seq_len), valid_tok_mask)
        u2 = self.apply_mean_mask(self.mlp_head_utt2(x).reshape(B, seq_len), valid_tok_mask)
        u3 = self.apply_mean_mask(self.mlp_head_utt3(x).reshape(B, seq_len), valid_tok_mask)
        u4 = self.apply_mean_mask(self.mlp_head_utt4(x).reshape(B, seq_len), valid_tok_mask)
        u5 = self.apply_mean_mask(self.mlp_head_utt5(x).reshape(B, seq_len), valid_tok_mask)

        return u1, u2, u3, u4, u5, p, w1, w2, w3

class BaselineLSTM2(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, input_dim=84):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.lstm = torch.nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=depth, batch_first=True)

        # for phone classification
        self.in_proj = nn.Linear(self.input_dim, embed_dim)
        self.mlp_head_phn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # for word classification
        self.mlp_head_word1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # phone projection
        self.phn_proj = nn.Linear(40, embed_dim)

        # utterance level
        self.mlp_head_utt1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt4 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt5 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

    # get the avg of only the valid token
    def apply_mean_mask(self, input, mask):
        input = input * mask
        input = torch.sum(input, dim=1)
        input = input / torch.sum(mask, dim=1)
        return input.unsqueeze(1)

    # x shape in [batch_size, sequence_len, feat_dim]
    # phn in [batch_size, seq_len]
    def forward(self, x, phn):

        # batch size
        B = x.shape[0]
        seq_len = x.shape[1]
        valid_tok_mask = (phn>=0)
        #print(valid_tok_mask.shape)

        # phn_one_hot in shape [batch_size, seq_len, feat_dim]
        phn_one_hot =  torch.nn.functional.one_hot(phn.long()+1, num_classes=40).float()
        # phn_embed in shape [batch_size, seq_len, embed_dim]
        phn_embed = self.phn_proj(phn_one_hot)

        if self.embed_dim != self.input_dim:
            x = self.in_proj(x)

        x = x + phn_embed

        x = self.lstm(x)[0]

        p = self.mlp_head_phn(x).reshape(B, seq_len, 1)

        w1 = self.mlp_head_word1(x).reshape(B, seq_len, 1)
        w2 = self.mlp_head_word2(x).reshape(B, seq_len, 1)
        w3 = self.mlp_head_word3(x).reshape(B, seq_len, 1)

        u1 = self.apply_mean_mask(self.mlp_head_utt1(x).reshape(B, seq_len), valid_tok_mask)
        u2 = self.apply_mean_mask(self.mlp_head_utt2(x).reshape(B, seq_len), valid_tok_mask)
        u3 = self.apply_mean_mask(self.mlp_head_utt3(x).reshape(B, seq_len), valid_tok_mask)
        u4 = self.apply_mean_mask(self.mlp_head_utt4(x).reshape(B, seq_len), valid_tok_mask)
        u5 = self.apply_mean_mask(self.mlp_head_utt5(x).reshape(B, seq_len), valid_tok_mask)

        return u1, u2, u3, u4, u5, p, w1, w2, w3

class BaselineLSTM(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, input_dim=84):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.lstm = torch.nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, num_layers=depth, batch_first=True)

        # for phone classification
        self.in_proj = nn.Linear(self.input_dim, embed_dim)
        self.mlp_head_phn = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # for word classification
        self.mlp_head_word1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_word3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

        # phone projection
        self.phn_proj = nn.Linear(40, embed_dim)

        # utterance level
        self.mlp_head_utt1 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt2 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt3 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt4 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.mlp_head_utt5 = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))

    # get the output of the last valid token
    def get_last_valid(self, input, mask):
        output = []
        B = input.shape[0]
        seq_len = input.shape[1]
        for i in range(B):
            for j in range(seq_len):
                if mask[i, j] == 0:
                    output.append(input[i, j-1])
                    break
                if j == seq_len - 1:
                    print('append')
                    output.append(input[i, j])
        output = torch.stack(output, dim=0)
        return output.unsqueeze(1)

    # x shape in [batch_size, sequence_len, feat_dim]
    # phn in [batch_size, seq_len]
    def forward(self, x, phn):

        # batch size
        B = x.shape[0]
        seq_len = x.shape[1]
        valid_tok_mask = (phn>=0)

        # phn_one_hot in shape [batch_size, seq_len, feat_dim]
        phn_one_hot =  torch.nn.functional.one_hot(phn.long()+1, num_classes=40).float()
        # phn_embed in shape [batch_size, seq_len, embed_dim]
        phn_embed = self.phn_proj(phn_one_hot)

        if self.embed_dim != self.input_dim:
            x = self.in_proj(x)

        x = x + phn_embed

        x = self.lstm(x)[0]

        p = self.mlp_head_phn(x).reshape(B, seq_len, 1)

        w1 = self.mlp_head_word1(x).reshape(B, seq_len, 1)
        w2 = self.mlp_head_word2(x).reshape(B, seq_len, 1)
        w3 = self.mlp_head_word3(x).reshape(B, seq_len, 1)

        u1 = self.get_last_valid(self.mlp_head_utt1(x).reshape(B, seq_len), valid_tok_mask)
        u2 = self.get_last_valid(self.mlp_head_utt2(x).reshape(B, seq_len), valid_tok_mask)
        u3 = self.get_last_valid(self.mlp_head_utt3(x).reshape(B, seq_len), valid_tok_mask)
        u4 = self.get_last_valid(self.mlp_head_utt4(x).reshape(B, seq_len), valid_tok_mask)
        u5 = self.get_last_valid(self.mlp_head_utt5(x).reshape(B, seq_len), valid_tok_mask)

        return u1, u2, u3, u4, u5, p, w1, w2, w3

if __name__ == '__main__':
    ast_mdl = BaselineLSTM(embed_dim=30, num_heads=1, depth=6)

    # input to AST should be [batch_size, sequence_len, embed_din]
    test_input = torch.rand([10, 50, 84])
    test_phn = torch.zeros([10, 50]).long()
    u1, u2, u3, u4, u5, p, w1, w2, w3 = ast_mdl(test_input, test_phn)
    print(u1.shape)
    print(p.shape)
    print(w1.shape)
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import gc
import transformer.Constants as Constants
from transformer.Layers import EncoderLayer
from tqdm import tqdm
torch.manual_seed(123)
from typing import List, Tuple, Union

def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask



class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out



class STAR(torch.nn.Module):
    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()
        
        
        
        self.d_model = d_model
        self.num_types = num_types

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))
        
        self.event_emb = nn.Embedding(num_types+1, d_model, padding_idx=Constants.PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])
            
        self.fusion_layer = nn.Linear(1024, 512)

 
        

    def temporal_enc(self, time, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = time.unsqueeze(-1) / self.position_vec

        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])

        return result * non_pad_mask
    
    def spatial_enc(self, xy, non_pad_mask):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """

        result = xy.unsqueeze(-1) / self.position_vec
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result * non_pad_mask
    

    def forward(self, event_type, event_time, xy,non_pad_mask):
           
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask_keypad = slf_attn_mask_keypad.type_as(slf_attn_mask_subseq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

                
        tem_enc = self.temporal_enc(event_time, non_pad_mask)
        temporal_input_embedded = self.event_emb(event_type)
        #print(tem_enc.size(), temporal_input_embedded.size(), 'aaaaaaaaaaaaa' )

        for enc_layer in self.layer_stack:
            temporal_input_embedded += tem_enc
            temporal_input_embedded, _ = enc_layer(
                temporal_input_embedded,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        #print(xy.size(), event_type.size(),event_time.size(), 'ddddddddddddddddd' )
        
        spatial_encc = self.spatial_enc(xy, non_pad_mask)
        spatial_input_embedded = self.event_emb(event_type)

        #print(temporal_input_embedded.size(), 'ddddddddddddddddd' )

            
        for enc_layer in self.layer_stack:
            spatial_input_embedded += spatial_encc
            spatial_input_embedded, xattn = enc_layer(
                spatial_input_embedded,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        

        spatial_input_embedded2 = spatial_input_embedded
        #temporal_input_embedded_last = self.temporal_encoder_1(temporal_input_embedded)#[-1]
        temporal_input_embedded_last = temporal_input_embedded[:,-334:,:]
        temporal_input_embedded = temporal_input_embedded[:,:-334,:]
        fusion_feat = torch.cat((temporal_input_embedded_last, spatial_input_embedded[:,-334:,:]), dim=2)
        #print(fusion_feat.size())
        fusion_feat = self.fusion_layer(fusion_feat)

        for enc_layer in self.layer_stack:
            spatial_input_embedded, _ = enc_layer(
                fusion_feat,
                non_pad_mask=non_pad_mask[:,-334:,:])        
        
        temporal_input_embedded = torch.cat((temporal_input_embedded, spatial_input_embedded), dim=1)

        #print(temporal_input_embedded.size())

        for enc_layer in self.layer_stack:
            outputs, _ = enc_layer(
                temporal_input_embedded,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)



        return outputs , xattn, spatial_input_embedded2

        
class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.encoder = STAR(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model, 1)

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)

    def forward(self, event_type, event_time, xy):
        """
        Return the hidden representations and predictions.
        For a sequence (l_1, l_2, ..., l_N), we predict (l_2, ..., l_N, l_{N+1}).
        Input: event_type: batch*seq_len;
               event_time: batch*seq_len.
        Output: enc_output: batch*seq_len*model_dim;
                type_prediction: batch*seq_len*num_classes (not normalized);
                time_prediction: batch*seq_len.
        """

            
        non_pad_mask = get_non_pad_mask(event_type)

        enc_output, attn , spatial_input_embedded2 = self.encoder(event_type, event_time, xy, non_pad_mask)
        

        time_prediction = self.time_predictor(enc_output, non_pad_mask)
        #print(time_prediction.size(), "l")
        type_prediction = self.type_predictor(enc_output, non_pad_mask)

        return enc_output, (type_prediction, time_prediction) , attn, spatial_input_embedded2

import logging
import math
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralhydrology.modelzoo.fc import FC

from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.modelzoo.head import get_head
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.utils.config import Config

LOGGER = logging.getLogger(__name__)


class FullTransformer(BaseModel):
    """Transformer model class, which relies on PyTorch's TransformerEncoder class.

    This class implements the encoder of a transformer network which can be used for regression.
    Unless the number of inputs is divisible by the number of transformer heads (``transformer_nheads``), it is
    necessary to use an embedding network that guarantees this. To achieve this, use ``statics/dynamics_embedding``,
    so the static/dynamic inputs will be passed through embedding networks before being concatenated. The embedding
    network will then map the static and dynamic features to size ``statics/dynamics_embedding['hiddens'][-1]``, so the
    total embedding size will be the sum of these values.
    The model configuration is specified in the config file using the following options:

    * ``transformer_positional_encoding_type``: choices to "sum" or "concatenate" positional encoding to other model
      inputs.
    * ``transformer_positional_dropout``: fraction of dropout applied to the positional encoding.
    * ``seq_length``: integer number of timesteps to treat in the input sequence.
    * ``transformer_nheads``: number of self-attention heads.
    * ``transformer_dim_feedforward``: dimension of the feed-forward networks between self-attention heads.
    * ``transformer_dropout``: dropout in the feedforward networks between self-attention heads.
    * ``transformer_nlayers``: number of stacked self-attention + feedforward layers.

    Parameters
    ----------
    cfg : Config
        The run configuration.
    """
    # specify submodules of the model that can later be used for finetuning. Names must match class attributes
    module_parts = ['embedding_net', 'transformer', 'head']

    def __init__(self, cfg: Config):
        super(FullTransformer, self).__init__(cfg=cfg)

        # embedding net before transformer
        self.embedding_net = InputLayer(cfg)

        # ensure that the number of inputs into the self-attention layer is divisible by the number of heads
        if self.embedding_net.output_size % cfg.transformer_nheads != 0:
            raise ValueError("Embedding dimension must be divisible by number of transformer heads. "
                             "Use statics_embedding/dynamics_embedding and embedding_hiddens to specify the embedding.")

        #if self.cfg.predict_last_n != self.cfg.seq_length - 1 and not self.cfg.is_eval:
            #raise ValueError("For the transformer model the length of predict_last_n needs to be seq_length - 1 during training")

        if self.cfg.is_eval and self.cfg.predict_last_n > 1:
            raise ValueError("The transformer model does not currently support predicting more than 1 day in the future")

        self._sqrt_embedding_dim = math.sqrt(self.embedding_net.output_size)




        self.target_embedding = FC(input_size=len(cfg.target_variables), hidden_sizes=[self.embedding_net.output_size])

        # positional encoder
        self._positional_encoding_type = cfg.transformer_positional_encoding_type
        if self._positional_encoding_type.lower() == 'concatenate':
            self.model_dim = self.embedding_net.output_size * 2
        elif self._positional_encoding_type.lower() == 'sum':
            self.model_dim = self.embedding_net.output_size
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {self._positional_encoding_type}")



        self.positional_encoder = _PositionalEncoding(embedding_dim=self.embedding_net.output_size,
                                                  dropout=cfg.transformer_positional_dropout,
                                                  position_type=cfg.transformer_positional_encoding_type,
                                                  max_len=cfg.seq_length)

        # positional mask
        self._srcmask = None
        self._tgtmask = None

        # encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.model_dim,
                                                    nhead=cfg.transformer_nheads,
                                                    dim_feedforward=cfg.transformer_dim_feedforward,
                                                    dropout=cfg.transformer_dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers,
                                             num_layers=cfg.transformer_nlayers,
                                             norm=None)

        decoder_layers = nn.TransformerDecoderLayer(d_model=self.model_dim,
                                                    nhead=cfg.transformer_nheads,
                                                    dim_feedforward=cfg.transformer_dim_feedforward,
                                                    dropout=cfg.transformer_dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layers,
                                             num_layers=cfg.transformer_nlayers,
                                             norm=None)



        self.transformer = nn.Transformer(d_model=self.model_dim,
                                          nhead=cfg.transformer_nheads,
                                          num_encoder_layers=cfg.transformer_nlayers,
                                          num_decoder_layers=cfg.transformer_nlayers,
                                          dim_feedforward=cfg.transformer_dim_feedforward,
                                          dropout=cfg.transformer_dropout)



        # head (instead of a decoder)
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=self.model_dim, n_out=self.output_size)

        # init weights and biases
        #self._reset_parameters()

    def _reset_parameters(self):
        # this initialization strategy was tested empirically but may not be the universally best strategy in all cases.
        initrange = 0.1
        for layer in self.transformer.encoder.layers:
            layer.linear1.weight.data.uniform_(-initrange, initrange)
            layer.linear1.bias.data.zero_()
            layer.linear2.weight.data.uniform_(-initrange, initrange)
            layer.linear2.bias.data.zero_()

        for decoder_layer in self.transformer.decoder.layers:
            decoder_layer.linear1.weight.data.uniform_(-initrange, initrange)
            decoder_layer.linear1.bias.data.zero_()
            decoder_layer.linear2.weight.data.uniform_(-initrange, initrange)
            decoder_layer.linear2.bias.data.zero_()

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on a transformer model without decoder.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.

        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
                - `y_hat`: model predictions of shape [batch size, sequence length, number of target variables].
        """
        # pass dynamic and static inputs through embedding layers, then concatenate them
        x_d = self.embedding_net(data)
        #x_d = x_d[1:]
        y = data['y']
        y = y.transpose(0, 1)
        y = y[:-1]
        #print(y.shape)
        #nanmask = ~torch.isnan(y)
        #print(nanmask.shape)
        y = y.nan_to_num()  #NOTE TO SELF, MASK NAN
        y = y.repeat(1,1,self.model_dim)
        #y = self.target_embedding(y)



        positional_encoding = self.positional_encoder(x_d * self._sqrt_embedding_dim)
        y_positional_encoding = self.positional_encoder(y * self._sqrt_embedding_dim)


        # mask out future values
        if self._srcmask is None or self._srcmask.size(0) != len(x_d):
            self._srcmask = torch.triu(x_d.new_full((len(x_d), len(x_d)), fill_value=float('-inf')), diagonal=1)
            self._srcmask = torch.roll(self._srcmask, shifts=1, dims=1)
            self._srcmask[:,0] = 0

        if self._tgtmask is None or self._tgtmask.size(0) != len(y):
            self._tgtmask = torch.triu(y.new_full((len(y), len(y)), fill_value=float('-inf')), diagonal=1)



        #print(self._tgtmask.shape)

        #self._tgtmask = self._tgtmask | nanmask

        #print(self._tgtmask)

        #self._tgtmask[:,-self.cfg.predict_last_n:]= float('-inf')



        #encoder_output = self.encoder(positional_encoding, self._srcmask)

        '''y_hat = y[:-self.cfg.predict_last_n,:,:]

        for i in range(self.cfg.seq_length - self.cfg.predict_last_n,self.cfg.seq_length):
            ones = torch.zeros(1, y.size()[1], y.size()[2]).to(torch.device(self.cfg.device))
            y_hat = torch.cat((y_hat, ones))

            self._srcmask = torch.triu(x_d.new_full((len(y_hat), len(x_d)), fill_value=float('-inf')), diagonal=1)

            self._tgtmask = torch.triu(y_hat.new_full((len(y_hat), len(y_hat)), fill_value=float('-inf')), diagonal=1)

            if(self.cfg.is_eval):
                # Transformer
                y_hat = self.decoder(y_hat,encoder_output,self._tgtmask,self._srcmask)
            else:
                input = y[:i+1,:,:]
                y_hat = self.decoder(y_hat, encoder_output, self._tgtmask, self._srcmask)
                #y_hat[i] = output[i]
            print(y_hat.shape)'''

        # Transformer
        output = self.transformer(src= positional_encoding, tgt = y_positional_encoding, tgt_mask = self._tgtmask, src_mask = self._srcmask)


        # head
        pred = self.head(self.dropout(output.transpose(0, 1)))
        # add embedding and positional encoding to output
        pred['embedding'] = x_d
        pred['positional_encoding'] = positional_encoding

        return pred


class _PositionalEncoding(nn.Module):
    """Class to create a positional encoding vector for timeseries inputs to a model without an explicit time dimension.

    This class implements a sin/cos type embedding vector with a specified maximum length. Adapted from the PyTorch
    example here: https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Parameters
    ----------
    embedding_dim : int
        Dimension of the model input, which is typically output of an embedding layer.
    dropout : float
        Dropout rate [0, 1) applied to the embedding vector.
    max_len : int, optional
        Maximum length of positional encoding. This must be larger than the largest sequence length in the sample.
    """

    def __init__(self, embedding_dim, position_type, dropout, max_len=5000):
        super(_PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, int(np.ceil(embedding_dim / 2) * 2))
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(max_len * 2) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe[:, :embedding_dim].unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        if position_type.lower() == 'concatenate':
            self._concatenate = True
        elif position_type.lower() == 'sum':
            self._concatenate = False
        else:
            raise RuntimeError(f"Unrecognized positional encoding type: {position_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for positional encoding. Either concatenates or adds positional encoding to encoder input data.

        Parameters
        ----------
        x : torch.Tensor
            Dimension is ``[sequence length, batch size, embedding output dimension]``.
            Data that is to be the input to a transformer encoder after including positional encoding.
            Typically this will be output from an embedding layer.

        Returns
        -------
        torch.Tensor
            Dimension is ``[sequence length, batch size, encoder input dimension]``.
            The encoder input dimension is either equal to the embedding output dimension (if ``position_type == sum``)
            or twice the embedding output dimension (if ``position_type == concatenate``).
            Encoder input augmented with positional encoding.

        """
        if self._concatenate:
            x = torch.cat((x, self.pe[:x.size(0), :].repeat(1, x.size(1), 1)), 2)
        else:
            x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

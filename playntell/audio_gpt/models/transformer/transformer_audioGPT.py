import copy

import torch
from torch import nn

from audio_gpt.models.containers import ModuleList
from audio_gpt.models.transformer.config import GPT2Config
from audio_gpt.models.transformer.gpt_decoder_audioGPT import GPT2LMHeadModel
from audio_gpt.models.transformer.load_gptmodel import load_weight
from ..captioning_model import CaptioningModel

state_dict = torch.load(
    "playntell/audio_gpt/gpt2-pytorch_model.bin",
    map_location="cpu" if not torch.cuda.is_available() else None,
)


class Transformer_audiogpt(CaptioningModel):
    def __init__(
        self, bos_idx, encoder, gpt2_type, n_layer=12, tau=0, srau=True, n_modes=1
    ):
        super(Transformer_audiogpt, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.gpt2_type = gpt2_type

        if gpt2_type == "random":

            config = GPT2Config()
            config.n_layer = n_layer
            config.srau = srau
            config.n_modes = n_modes
            decoder = GPT2LMHeadModel(config, tau=tau)

            self.decoder = decoder

        else:

            config = GPT2Config()
            config.n_layer = n_layer
            config.srau = srau
            config.n_modes = n_modes
            decoder = GPT2LMHeadModel(config, tau=tau)

            decoder = load_weight(decoder, state_dict)

            self.decoder = decoder

        self.register_state("enc_output", None)
        self.register_state("mask_enc", None)
        self.init_weights()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):

        if self.gpt2_type == "random":
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            for p in self.encoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, images, seq, *args):
        enc_output, mask_enc = self.encoder(images)

        dec_output, past, enc_dec_attn_masks = self.decoder(seq, enc_output, mask_enc)
        return dec_output, past, enc_dec_attn_masks

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device), None, None]

    def step(self, t, prev_output, visual, seq, past, mode="teacher_forcing", **kwargs):
        it = None
        if mode == "teacher_forcing":
            raise NotImplementedError
        elif mode == "feedback":
            if t == 0:
                self.enc_output, self.mask_enc = self.encoder(visual)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = (
                        visual[0]
                        .data.new_full((visual[0].shape[0], 1), self.bos_idx)
                        .long()
                    )
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc, past=past)

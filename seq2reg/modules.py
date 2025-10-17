import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from flash_attn.modules.mha import MHA
from flash_attn.bert_padding import (
    pad_input,
    unpad_input,
)


# From https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742
def get_alibi_slopes(n):
    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return torch.tensor(
            get_slopes_power_of_2(n)
        )  # In the paper, we only train models that have 2^a heads for some a. This function has

    else:  # some good properties that only occur when the input is a power of 2. To maintain that even
        closest_power_of_2 = 2 ** math.floor(
            math.log2(n)
        )  # when the number of heads is not a power of 2, we use this workaround.
        return torch.tensor(
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][
                : n - closest_power_of_2
            ].tolist()
        )


def get_relative_positions(seq_len: int) -> torch.tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return x - y


class ContextFlashAttentionEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        hidden_dim=2048,
        dropout=0.1,
        batch_first=True,
        use_alibi=False,
        make_data_kv=False,
        mlp_dout=0.0,
    ):
        super().__init__()
        self.mixer = FlashAttLayer(
            d_model, nhead, dropout=dropout, use_alibi=use_alibi, cross_attn=False
        )
        self.crossMHA = FlashAttLayer(
            d_model, nhead, dropout=dropout, use_alibi=False, cross_attn=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.linear_geglu_1 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(mlp_dout)
        self.linear_geglu_2 = nn.Linear(hidden_dim // 2, d_model)
        self.use_alibi = use_alibi
        self.num_heads = nhead
        self.make_data_kv = make_data_kv
        if use_alibi:
            self.register_buffer("m", get_alibi_slopes(self.num_heads))

    def forward(self, src, context, key_padding_mask=None, precision=torch.float32):
        res_long = src
        res_short = src
        dtype = src.dtype

        flash_dtype = torch.float16 if precision == torch.float32 else precision

        assert src.shape == context.shape, "src and context must have the same shape"
        x = self.norm1(src)
        qkv = x

        if flash_dtype is not None:
            # flash-att only support 16-bit floating point
            self.mixer.to(flash_dtype)
            qkv = qkv.to(flash_dtype)

        x = self.mixer(qkv, src_key_padding_mask=key_padding_mask, precision=precision)
        if flash_dtype is not None:
            self.mixer.to(src.dtype)
            x = x.to(src.dtype)
        x += res_short
        res_short = x

        x = self.norm2(x)
        if self.make_data_kv:
            q, kv = context, x
        else:
            q, kv = x, context
        if flash_dtype is not None:
            self.crossMHA.to(flash_dtype)
            q = q.to(flash_dtype)
            kv = kv.to(flash_dtype)
        # self.crossMHA.to(flash_dtype)

        x = self.crossMHA(
            q,
            kv,
            src_key_padding_mask=key_padding_mask,
            context_key_padding_mask=key_padding_mask,
            precision=precision,
        )
        if flash_dtype is not None:
            x = x.to(src.dtype)
            self.crossMHA.to(src.dtype)
        x += res_short

        x = self.norm3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.dropout(x)
        x = self.linear_geglu_2(x)
        x += res_long

        return x


class FlashTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        hidden_dim=2048,
        dropout=0.1,
        use_alibi=False,
        mlp_dout=0.1,
    ):
        super().__init__()
        self.MHA = MHA(
            d_model, nhead, dropout=dropout, use_flash_attn=True, use_alibi=use_alibi
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear_geglu_1 = nn.Linear(d_model, hidden_dim)
        self.dropout = nn.Dropout(mlp_dout)
        self.linear_geglu_2 = nn.Linear(hidden_dim // 2, d_model)

    def forward(self, src, src_key_padding_mask=None, precision=torch.float32):
        dtype = src.dtype
        flash_dtype = torch.float16 if precision == torch.float32 else precision
        res_long = src
        res_short = src

        src = self.norm1(src)
        batch, seqlen = src.shape[:2]
        if src_key_padding_mask is not None:
            src_key_padding_mask = ~src_key_padding_mask
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch, _ = unpad_input(
                src, src_key_padding_mask
            )
            mixer_kwargs = {"cu_seqlens": cu_seqlens, "max_seqlen": max_seqlen_in_batch}
            if flash_dtype is not None:
                hidden_states = hidden_states.to(flash_dtype)
                self.MHA.to(flash_dtype)

            hidden_states = self.MHA(hidden_states, **mixer_kwargs)

            if flash_dtype is not None:
                self.MHA.to(dtype)
            hidden_states = pad_input(hidden_states, indices, batch, seqlen)

            if flash_dtype is not None:
                hidden_states = hidden_states.to(dtype)
                src_key_padding_mask = src_key_padding_mask.to(dtype)

            x = hidden_states
            x *= src_key_padding_mask.unsqueeze(-1)  # zero out padding
        else:
            if flash_dtype is not None:
                src = src.to(flash_dtype)
            x = self.MHA(src)
        x += res_short
        x = self.norm2(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        x = x * F.gelu(gate)
        x = self.dropout(x)
        x = self.linear_geglu_2(x)
        x += res_long

        return x


class FlashAttLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        hidden_dim=2048,
        dropout=0.1,
        use_alibi=False,
        cross_attn=False,
    ):
        super().__init__()
        self.cross_attn = cross_attn
        self.MHA = MHA(
            d_model,
            nhead,
            dropout=dropout,
            use_flash_attn=True,
            use_alibi=use_alibi,
            cross_attn=cross_attn,
        )

    def forward(
        self,
        src,
        cntx=None,
        src_key_padding_mask=None,
        context_key_padding_mask=None,
        precision=torch.float32,
    ):
        dtype = src.dtype
        flash_dtype = torch.float16 if precision == torch.float32 else precision
        x_short = src
        batch, seqlen = src.shape[:2]
        mixer_kwargs = {}
        if self.cross_attn:
            assert cntx is not None
            if src_key_padding_mask is not None:
                assert (
                    context_key_padding_mask is not None
                ), "context_key_padding_mask must be provided if src_key_padding_mask is provided"
                src_key_padding_mask = ~src_key_padding_mask
                (
                    src_hidden_states,
                    src_indices,
                    src_cu_seqlens,
                    src_max_seqlen_in_batch,
                    _,
                ) = unpad_input(src, src_key_padding_mask)
                mixer_kwargs = {
                    "cu_seqlens": src_cu_seqlens,
                    "max_seqlen": src_max_seqlen_in_batch,
                }
            else:
                src_hidden_states = src

            if context_key_padding_mask is not None:
                assert (
                    src_key_padding_mask is not None
                ), "src_key_padding_mask must be provided if context_key_padding_mask is provided"
                context_key_padding_mask = ~context_key_padding_mask
                (
                    cntx_hidden_states,
                    cntx_indices,
                    cntx_cu_seqlens,
                    cntx_max_seqlen_in_batch,
                    _,
                ) = unpad_input(cntx, context_key_padding_mask)
                mixer_kwargs.update(
                    {
                        "cu_seqlens_k": cntx_cu_seqlens,
                        "max_seqlen_k": cntx_max_seqlen_in_batch,
                    }
                )
            else:
                cntx_hidden_states = cntx
            if flash_dtype is not None:
                src_hidden_states = src_hidden_states.to(flash_dtype)
                cntx_hidden_states = cntx_hidden_states.to(flash_dtype)
            hidden_states = self.MHA(
                src_hidden_states, cntx_hidden_states, **mixer_kwargs
            )

            if src_key_padding_mask is not None:
                hidden_states = pad_input(hidden_states, src_indices, batch, seqlen)

            x = hidden_states

        else:
            if src_key_padding_mask is not None:
                src_key_padding_mask = ~src_key_padding_mask
                hidden_states, indices, cu_seqlens, max_seqlen_in_batch, _ = (
                    unpad_input(src, src_key_padding_mask)
                )
                mixer_kwargs = {
                    "cu_seqlens": cu_seqlens,
                    "max_seqlen": max_seqlen_in_batch,
                }
                if flash_dtype is not None:
                    hidden_states = hidden_states.to(flash_dtype)
                hidden_states = self.MHA(hidden_states, **mixer_kwargs)
                hidden_states = pad_input(hidden_states, indices, batch, seqlen)
                x = hidden_states

            else:
                x = self.MHA(src)
        return x

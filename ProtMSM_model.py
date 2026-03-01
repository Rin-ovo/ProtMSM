import inspect
import math
from typing import  Optional, Tuple, Union
import torch.nn.functional as F
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    SequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    logging,
)
from configuration_model import Config

# try except block so it'll work with trust_remote_code.
# Based on Jamba architecture by AI21 Labs and Transformers by HuggingFace.
# Includes kernels from mamba-ssm (Copyright Dao-AILab).
# try except block so it'll work with trust_remote_code.
try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None


is_fast_path_available = all(
    (selective_state_update, selective_scan_fn, mamba_inner_fn)
)

logger = logging.get_logger(__name__)



class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6,device=None):
        """
        JambaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size,device=device))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        hidden_states = hidden_states.to(self.weight.device)
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:

    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def chunk_pooling(x, target_len: int):

    B, L, H = x.shape
    if L == target_len:
        return x
    if L < target_len:
        # padding
        pad_len = target_len - L
        pad = x.new_zeros(B, pad_len, H)
        return torch.cat([x, pad], dim=1)


    chunk_size = math.ceil(L / target_len)
    chunks = []
    for i in range(0, L, chunk_size):
        segment = x[:, i:i+chunk_size, :]
        pooled = segment.mean(dim=1, keepdim=True)
        chunks.append(pooled)
    pooled_x = torch.cat(chunks, dim=1)
    return pooled_x[:, :target_len, :]


import torch
import torch.nn as nn
from typing import Optional


class SimpleSdpaAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads


        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )


        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        bsz, q_len, _ = hidden_states.shape
        attention_mask = (attention_mask == 1).bool()


        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)


        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)


        key_states = key_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)
        value_states = value_states.repeat_interleave(self.num_heads // self.num_key_value_heads, dim=1)


        if query_states.device.type == "cuda":
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        if attention_mask is not None:

            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            attention_mask = attention_mask.to(query_states.dtype)
            attention_mask = (1.0 - attention_mask) * -1e9

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask
        )


        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)
        output = self.o_proj(attn_output)

        return output



class SSM(nn.Module):
    """
    Compute ∆, A, B, C, and D the state space parameters and compute the `contextualized_states`.
    A, D are input independent (see Mamba paper [1] Section 3.5.2 "Interpretation of A" for why A isn't selective)
    ∆, B, C are input-dependent (this is a key difference between Mamba and the linear time invariant S4,
    and is why Mamba is called **selective** state spaces)
    """


    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.mamba_d_state
        self.conv_kernel_size = config.mamba_d_conv
        self.intermediate_size = config.mamba_expand * config.hidden_size
        self.time_step_rank = config.mamba_dt_rank
        self.use_conv_bias = config.mamba_conv_bias
        self.use_bias = config.mamba_proj_bias


        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding="same",
        )

        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.use_fast_kernels = config.use_mamba_kernels

        # projection of the input hidden states
        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=self.use_bias)
        # selective projection used to make dt, B and C input dependant
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        # time step projection (discretization)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        # S4D real initialization. These are not discretized!
        # The core is to load them, compute the discrete states, then write the updated state. Keeps the memory bounded

        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)

        self.dt_layernorm = RMSNorm(self.time_step_rank, eps=config.rms_norm_eps)
        self.b_layernorm = RMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)
        self.c_layernorm = RMSNorm(self.ssm_state_size, eps=config.rms_norm_eps)

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because on of `(selective_state_update, selective_scan_fn,mamba_inner_fn)`"
                " is None. To install follow https://github.com/state-spaces/mamba/#installation and"
                " https://github.com/Dao-AILab/causal-conv1d. If you want to use the naive implementation, set `use_mamba_kernels=False` in the model config"
            )

    def cuda_kernels_forward(self, hidden_states: torch.Tensor):
        batch_size, seq_len, _ = hidden_states.shape


        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states).transpose(1, 2)

        # We can't use `mamba_inner_fn` even if in training and without cache params because we have the
        # inner layernorms which isn't supported by this fused kernel
        hidden_states, gate = projected_states.chunk(2, dim=1)  # 拆分为两部分


        hidden_states = self.conv1d(input=hidden_states)
        hidden_states = self.act(hidden_states)

        # 3. State Space Model sequence transformation
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )


        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)


        time_proj_bias = self.dt_proj.bias
        self.dt_proj.bias = None
        discrete_time_step = self.dt_proj(time_step).transpose(1, 2)
        self.dt_proj.bias = time_proj_bias


        A = -torch.exp(self.A_log.float()).to(hidden_states.device)

        time_proj_bias = time_proj_bias.float() if time_proj_bias is not None else None


        scan_outputs, ssm_state = selective_scan_fn(
            hidden_states.to(dtype=torch.float32, device=hidden_states.device),
            discrete_time_step.to(dtype=torch.float32, device=hidden_states.device),
            A.to(dtype=torch.float32, device=hidden_states.device),
            B.transpose(1, 2).to(dtype=torch.float32, device=hidden_states.device),
            C.transpose(1, 2).to(dtype=torch.float32, device=hidden_states.device),
            self.D.to(dtype=torch.float32, device=hidden_states.device),
            gate.to(dtype=torch.float32, device=hidden_states.device),
            time_proj_bias.to(dtype=torch.float32, device=hidden_states.device) if time_proj_bias is not None else None,
            delta_softplus=True,
            return_last_state=True,
        )

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))
        return contextualized_states

    def slow_forward(self, input_states):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype


        # 1. Gated MLP's linear projection
        projected_states = self.in_proj(input_states).transpose(1, 2)  # [batch, 2 * intermediate_size, seq_len]
        hidden_states, gate = projected_states.chunk(2, dim=1)

        # 2. Convolution sequence transformation
        ssm_state = torch.zeros(
            (batch_size, self.intermediate_size, self.ssm_state_size),
            device=hidden_states.device, dtype=torch.float32
        )


        hidden_states = self.act(self.conv1d(hidden_states))  # [batch, intermediate_size, seq_len]

        # 3. State Space Model sequence transformation
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )

        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)

        discrete_time_step = self.dt_proj(time_step)
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1,2)


        # 3.b. Discretization: B and C to [batch, seq_len, intermediate_size, ssm_state_size] (SRAM)
        A = -torch.exp(self.A_log.to(dtype=torch.float32, device=hidden_states.device))
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :,None])
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].to(dtype=torch.float32,device=hidden_states.device)
        deltaB_u = discrete_B * hidden_states[:, :, :, None].to(dtype=torch.float32)

        # 3.c perform the recurrence y ← SSM(A, B, C)(x)
        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i,:]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        # 4. Final linear projection
        contextualized_states = self.out_proj(scan_output.transpose(1, 2))
        return contextualized_states

    def forward(self, hidden_states):
        if self.use_fast_kernels:
            if not is_fast_path_available or "cuda" not in self.x_proj.weight.device.type:
                raise ValueError(
                    "Fast Mamba kernels are not available. Make sure to they are installed and that the mamba module is on a CUDA device"
                )

            return self.cuda_kernels_forward(hidden_states)
        return self.slow_forward(hidden_states)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]


    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))




class JambaPreTrainedModel(PreTrainedModel):
    config_class = Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True



    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()





class LongParallelLayer(nn.Module):



    def __init__(self, config,ssm_pool_size=768):
        super().__init__()

        self.attention = SimpleSdpaAttention(config)


        self.ssm1 = SSM(config)
        self.ssm2 = SSM(config)


        self.mix_weight = nn.Parameter(torch.tensor(0.5))

        # LayerNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


        # Feed-forward
        self.feed_forward = MLP(config)

        self.ssm_pool_size = ssm_pool_size

    def forward(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)


        attn_input = hidden_states
        attn_output = self.attention(
            attn_input, attention_mask
        )

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            summed = torch.sum(attn_output * mask, dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            attn_pooled = summed / counts
        else:
            attn_pooled = attn_output.mean(dim=1)

        ssm_input = chunk_pooling(hidden_states, self.ssm_pool_size)

        ssm_output1 = self.ssm1(ssm_input)
        ssm_output2 = self.ssm2(ssm_output1)

        ssm_pooled = ssm_output2.mean(dim=1)


        alpha = torch.sigmoid(self.mix_weight)
        parallel_output = alpha * attn_pooled + (1 - alpha) * ssm_pooled

        return parallel_output

class ShortFeatureExtractor(nn.Module):

    def __init__(self, hidden_size, num_heads=8, kernel_sizes=(3, 7, 15), dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)


        self.convs = nn.ModuleList([
            nn.Conv1d(hidden_size, hidden_size, kernel_size=k, padding="same", groups=1, bias=False)
            for k in kernel_sizes
        ])
        self.act = nn.GELU()


        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)


        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)


        self.fusion_weight = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, attn_mask=None):
        """
        x: (B, L, H)
        attn_mask: (B, L)  -> 1=keep, 0=pad
        """


        residual = x
        conv_outs = []
        x_t = x.transpose(1, 2)
        for conv in self.convs:
            conv_outs.append(self.act(conv(x_t)))
        x_conv = sum(conv_outs).transpose(1, 2)
        x = residual + self.dropout(x_conv)
        x = self.norm1(x)


        residual = x
        key_padding_mask = None
        if attn_mask is not None:
            key_padding_mask = attn_mask == 0
        x_attn, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)


        w = torch.sigmoid(self.fusion_weight).view(1, 1, -1)
        fused = w * x_attn + (1 - w) * x
        x = residual + self.dropout(fused)
        x = self.norm2(x)

        if attn_mask is not None:
            mask = attn_mask.unsqueeze(-1)
            summed = torch.sum(x * mask, dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            protein_feature = summed / counts
        else:
            protein_feature = x.mean(dim=1)

        return protein_feature



class Classification(JambaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.model = LongParallelLayer(config)
        esm_dim =  1280
        self.esm_proj = nn.Linear(esm_dim, config.hidden_size, bias=False)
        self.num_labels = config.num_labels
        self.score = nn.Linear(self.config.hidden_size, self.num_labels, bias=False)

        self.short_extractor = ShortFeatureExtractor(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            kernel_sizes=(3, 7, 15)
        )

        self.gate_k = nn.Parameter(torch.tensor(0.01))
        threshold = getattr(config, "length_threshold", 1024.0)
        self.gate_b = nn.Parameter(torch.tensor(float(threshold)))

        self.post_init()

    def forward(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None


    ) -> Union[Tuple, SequenceClassifierOutput]:

        if inputs_embeds is None:
            raise ValueError("Classification requires inputs_embeds (token-level features, e.g. ESM outputs)")


        proj = self.esm_proj(inputs_embeds)

        B, L, H = proj.shape

        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).float()
        else:
            lengths = torch.full((B,), float(L), device=proj.device, dtype=proj.dtype)

        gate = torch.sigmoid(self.gate_k * (lengths - self.gate_b))
        gate = gate.view(B, 1)

        Long_outputs = self.model(
            hidden_states=proj,
            attention_mask=attention_mask

        )


        long_hidden = Long_outputs[0]


        short_hidden = proj
        short_hidden = self.short_extractor(short_hidden, attention_mask)

        gate = gate.view(B, 1)
        blended = gate * long_hidden + (1.0 - gate) * short_hidden

        protein_repr = blended


        logits = self.score(protein_repr)

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels.float())
            return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=blended
            )
        else:
            return SequenceClassifierOutput(
                logits=logits,
                hidden_states=blended
            )

# coding=utf-8
# Copyright 2024 AI21 Labs Ltd. and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications Copyright 2026.
# This file has been modified from the original Jamba configuration to support
# protein sequence analysis parameters (length_threshold, ontology).
import math
from utils import get_go_ic
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class Config(PretrainedConfig):



    def __init__(
            self,
            hidden_size=512,
            intermediate_size=1024,
            num_hidden_layers=3,
            num_attention_heads=8,
            num_key_value_heads=2,
            hidden_act="silu",
            initializer_range=0.02,
            rms_norm_eps=1e-6,
            attention_dropout=0.0,
            use_mamba_kernels=True,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_dt_rank="auto",
            mamba_conv_bias=True,
            mamba_proj_bias=False,
            length_threshold=1024,
            ontology='bp',
            **kwargs,
    ):

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.attention_dropout = attention_dropout
        self.length_threshold = length_threshold


        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps



        self.use_mamba_kernels = use_mamba_kernels
        self.mamba_d_state = mamba_d_state
        self.mamba_d_conv = mamba_d_conv
        self.mamba_expand = mamba_expand
        self.mamba_dt_rank = math.ceil(self.hidden_size / 16) if mamba_dt_rank == "auto" else mamba_dt_rank
        self.mamba_conv_bias = mamba_conv_bias
        self.mamba_proj_bias = mamba_proj_bias
        self.ontology = ontology
        self.num_labels = self.load_num_label(self.ontology)




        super().__init__(
            num_labels=self.num_labels,
            **kwargs,
        )

    def load_num_label(self,ont):
        _,go_list = get_go_ic(f'data/{ont}/{ont}_go_ic.txt')
        return len(go_list)


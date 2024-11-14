import os
import json
import logging
import torch.distributed
from transformers import AutoModelForCausalLM, AutoModel, PreTrainedModel, PretrainedConfig, Qwen2ForCausalLM, GenerationConfig

import torch
from torch import nn

from .utils import prepare_inputs_for_generation_with_noncontinuous_positions
logger = logging.getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, slm_dim, llm_dim, dim_reduction=False):
        super(MLP, self).__init__()
        if dim_reduction:
            self.linear1 = nn.Linear(llm_dim, slm_dim * 2)
            self.linear2 = nn.Linear(slm_dim * 2, slm_dim)
        else:
            self.linear1 = nn.Linear(slm_dim, slm_dim * 2)
            self.linear2 = nn.Linear(slm_dim * 2, llm_dim)
        self.relu = nn.ReLU()

    def forward(self, mt_hidden_state):
        output = self.linear1(mt_hidden_state)
        output = self.relu(output)
        output = self.linear2(output)
        return output

class Mapping(nn.Module):
    def __init__(self, slm_dim, llm_dim, dim_reduction=False):
        super(Mapping, self).__init__()
        self.dim_reduction = dim_reduction
        self.mlp = MLP(slm_dim, llm_dim, dim_reduction=dim_reduction)
        if dim_reduction:
            self.end_boundary = nn.Parameter(
                torch.zeros(1, 1, slm_dim), requires_grad=True
            )
        else:
            self.end_boundary = nn.Parameter(
                torch.zeros(1, 1, llm_dim), requires_grad=True
            )

    def forward(self, hidden_states):
        hidden_states = self.mlp(hidden_states)
        return hidden_states

    def get_embed(self):
        return self.end_boundary

class ItbConfig(PretrainedConfig):
    model_type = "transbridge"
    is_composition = False
    def __init__(
        self,
        llm_path=None,
        slm_path_a=None,
        slm_path_b=None,
        ignore_index=-100,
        training_stage=0,
        max_gen_len=300,
        itb_pad_token_id=0,
        itb_bos_token_id=1,
        **kwargs,
    ):  
        self.ignore_index = ignore_index
        self.training_stage = training_stage
        self.llm_path = llm_path
        self.slm_path_a = slm_path_a
        self.slm_path_b = slm_path_b
        self.max_gen_len = max_gen_len
        self.itb_bos_token_id = itb_bos_token_id
        self.itb_pad_token_id = itb_pad_token_id
        super().__init__(**kwargs)

class ImplicitTransBridge(PreTrainedModel):
    def __init__(self, config: ItbConfig):
        super(ImplicitTransBridge, self).__init__(config)
        self.config = config

        if self.config.training_stage == 0:
            llm = AutoModelForCausalLM.from_pretrained(self.config.llm_path)#, torch_dtype=torch.bfloat16)
        else:
            llm = AutoModel.from_pretrained(self.config.llm_path)

        self.llm = llm
        self.llm_embedding_layer = self.llm.get_input_embeddings()
        for name, parameter in self.llm.named_parameters():
            parameter.requires_grad = False

        if self.config.training_stage in [0,2]:  # 0: xx-en    2:  xx-xx
            self.mt_model = AutoModel.from_pretrained(self.config.slm_path_a)
            logger.info(f'Small LM A model size: {sum(param.numel() for param in self.mt_model.parameters()) / 1000000} M')
            self.slm_a = self.mt_model.get_encoder()
            for name, parameter in self.mt_model.named_parameters():
                parameter.data = parameter.data.contiguous()
                parameter.requires_grad = False
            self.mapping_a = Mapping(self.slm_a.config.d_model, llm.config.hidden_size)
            logger.info(f'mapping a layer size: {sum(param.numel() for param in self.mapping_a.parameters()) / 1000000} M')

        if self.config.training_stage in [1,2]:  # 1: en-xx    2:  xx-xx
            slm_b = AutoModelForCausalLM.from_pretrained(self.config.slm_path_b)
            logger.info(f'Small LM B model size: {sum(param.numel() for param in slm_b.parameters()) / 1000000} M')
            self.slm_b = slm_b
            for name, parameter in self.slm_b.named_parameters():
                parameter.requires_grad = True
            self.mapping_b = Mapping(slm_b.config.hidden_size, llm.config.hidden_size, dim_reduction=True)
            logger.info(f'mapping b layer size: {sum(param.numel() for param in self.mapping_b.parameters()) / 1000000} M')
            self.slm_b_embedding_layer = self.slm_b.get_input_embeddings()

        self.itb_pad_token_id = self.config.itb_pad_token_id
        self.itb_bos_token_id = self.config.itb_bos_token_id
        Qwen2ForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_with_noncontinuous_positions

    def save_pretrained(self, output_dir, state_dict=None, **kwargs):
        self.config.save_pretrained(output_dir)
        state_dict = {k: v for k, v in self.state_dict().items() if "mapping" in k or 'slm_b' in k}
        torch.save(state_dict, os.path.join(output_dir, f"mappings.pt"))

    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        config = ItbConfig.from_pretrained(model_path)
        model = cls(config)
        state_dict = torch.load(os.path.join(model_path, f"mappings.pt"))
        model.load_state_dict(state_dict, strict=False)
        return model

    def stage0_forward(self, input_ids_prefix, attention_mask_prefix,
                      input_ids_affix, attention_mask_affix,
                      input_ids_query=None, attention_mask_query=None,
                      input_ids_response=None, attention_mask_response=None):

        end_boundary = self.mapping_a.get_embed()
        bs = input_ids_prefix.size(0)
        end_boundary = end_boundary.expand([bs, 1, end_boundary.size(-1)])
        prefix_embedding = self.llm_embedding_layer(input_ids_prefix)
        prefix_mask = attention_mask_prefix
        mask = torch.ones([bs, 1], dtype=torch.long).cuda()

        slma_outputs = self.slm_a(input_ids=input_ids_query,
                                        attention_mask=attention_mask_query,
                                        output_hidden_states=True)
        slma_last_hidden_state = slma_outputs[0]
        slma_hidden_state = self.mapping_a(slma_last_hidden_state)
        llm_input_embedding = torch.cat([prefix_embedding, slma_hidden_state, end_boundary],
                                        dim=1)
        llm_input_mask = torch.cat([prefix_mask, attention_mask_query, mask], dim=1)

        if input_ids_affix is not None:
            hidden_states_affix = self.llm_embedding_layer(input_ids_affix)
            llm_input_embedding = torch.cat([llm_input_embedding, hidden_states_affix], dim=1)
            llm_input_mask = torch.cat([llm_input_mask, attention_mask_affix], dim=1)

        labels = None
        if input_ids_response is not None:
            pad_labels = llm_input_mask * -100 + (1 - llm_input_mask) * -100
            label_embedding = self.llm_embedding_layer(input_ids_response)
            llm_input_embedding = torch.cat([llm_input_embedding, label_embedding], dim=1)
            llm_input_mask = torch.cat([llm_input_mask, attention_mask_response], dim=1)
            labels = input_ids_response * attention_mask_response - 100 * (1 - attention_mask_response)
            labels = torch.cat([pad_labels, labels], dim=1)

        llm_input_position_ids = torch.cumsum(llm_input_mask, dim=1) - 1

        if labels is None:
            generate_ids = self.llm.generate(inputs_embeds=llm_input_embedding,
                                                   attention_mask=llm_input_mask,
                                                   position_ids=llm_input_position_ids,
                                                   #max_new_tokens=self.config.max_gen_len,
                                                   pad_token_id=self.itb_pad_token_id,
                                                   generation_config=GenerationConfig(
                                                        max_new_tokens=600,
                                                        do_sample=False,
                                                        temperature=0.0,  # t=0.0 raise error if do_sample=True
                                                    )
                                            )
            return generate_ids
        else:
            output = self.llm(inputs_embeds=llm_input_embedding,
                                    attention_mask=llm_input_mask,
                                    position_ids=llm_input_position_ids,
                                    labels=labels)
            return output

    def forward(self, input_ids_prefix, attention_mask_prefix,
                      input_ids_affix, attention_mask_affix,
                      input_ids_query=None, attention_mask_query=None,
                      input_ids_response=None, attention_mask_response=None):

        input_ids_prefix = input_ids_prefix.to(self.llm.device)
        attention_mask_prefix = attention_mask_prefix.to(self.llm.device)
        input_ids_affix = input_ids_affix.to(self.llm.device)
        attention_mask_affix = attention_mask_affix.to(self.llm.device)
        input_ids_query = input_ids_query.to(self.llm.device)
        attention_mask_query = attention_mask_query.to(self.llm.device)

        if input_ids_response is not None:
            input_ids_response = input_ids_response.to(self.llm.device)
            attention_mask_response = attention_mask_response.to(self.llm.device)

        if self.config.training_stage == 0:
            return self.stage0_forward(input_ids_prefix, attention_mask_prefix,
                                       input_ids_affix, attention_mask_affix,
                                       input_ids_query, attention_mask_query,
                                       input_ids_response, attention_mask_response)

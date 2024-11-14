import os
from transformers import TrainerCallback, Qwen2ForCausalLM
from transformers.cache_utils import StaticCache
from transformers.models.qwen2.modeling_qwen2 import _prepare_4d_causal_attention_mask_with_cache_position
import torch

class SaveModelCallback(TrainerCallback):
    def __init__(self, save_path):
        self.save_path = save_path

    def on_save(self, args, state, control, **kwargs):
        output_dir = os.path.join(self.save_path, f"step_{state.global_step}")
        kwargs["tokenizer"].save_pretrained(output_dir)
        print(f"Step {state.global_step}: Saving model to {output_dir}")
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(kwargs["model"])
        else:
            state_dict = self.model.state_dict()
        state_dict = {k: v  for k, v in state_dict.items() if "mapping" in k}
        torch.save(state_dict, os.path.join(output_dir, f"mappings.pt"))
        control.should_save = False

def prepare_inputs_for_generation_with_noncontinuous_positions(
    self: Qwen2ForCausalLM,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
    # Exception 1: when passing input_embeds, input_ids may be missing entries
    # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
    if past_key_values is not None:
        if inputs_embeds is not None:  # Exception 1
            input_ids = input_ids[:, -cache_position.shape[0] :]
        elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
            input_ids = input_ids[:, cache_position]
    if attention_mask is not None and (position_ids is None or position_ids.shape != input_ids.shape):
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1] :]
            # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
            position_ids = position_ids.clone(memory_format=torch.contiguous_format)
    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
    else:
        # The clone here is for the same reason as for `position_ids`.
        model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs["inputs_embeds"] is not None:
            batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
            device = model_inputs["inputs_embeds"].device
        else:
            batch_size, sequence_length = model_inputs["input_ids"].shape
            device = model_inputs["input_ids"].device
        dtype = self.lm_head.weight.dtype
        min_dtype = torch.finfo(dtype).min
        attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=past_key_values.get_max_length(),
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=batch_size,
        )

    model_inputs.update(
        {
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }
    )
    return model_inputs
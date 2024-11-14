import logging
from transformers import AutoTokenizer
from .modeling_itb import ImplicitTransBridge, ItbConfig
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def setup_model_and_tokenizer(model_args, training_args):
    tokenizer_llm = AutoTokenizer.from_pretrained(model_args.llm_path)
    tokenizer_slm = AutoTokenizer.from_pretrained(model_args.slm_path_a)
    if tokenizer_llm.bos_token_id is None:
        tokenizer_llm.add_special_tokens({"bos_token": "<bos>"})

    config = ItbConfig(
        llm_path=model_args.llm_path,
        slm_path_a=model_args.slm_path_a,
        slm_path_b=model_args.slm_path_b,
        ignore_index=-100,
        training_stage=model_args.stage,
        max_gen_len=model_args.max_gen_len,
        itb_bos_token_id=tokenizer_llm.bos_token_id,
        itb_pad_token_id=tokenizer_llm.pad_token_id,
    )
    model = ImplicitTransBridge(
        config=config,
    )

    return model, tokenizer_llm, tokenizer_slm

def count_parameters(model):
    r"""
    Returns the number of trainable parameters and number of all parameters in the model.
    """
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes, multiply the number of parameters by 2
        if param.__class__.__name__ == "Params4bit":
            num_params = num_params * 2

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param
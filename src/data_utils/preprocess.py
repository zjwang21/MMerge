from datasets import load_dataset

from .utils import translation_prompt
from ..utils.logging_utils import get_logger

logger = get_logger(__name__)

def preprocess_parallel_dataset(tokenizer, data_args, training_args):
    dataset = load_dataset("json", data_files=data_args.dataset_name)
    prompt = translation_prompt.format(data_args.src_lang, data_args.tgt_lang)
    prompt_ids = tokenizer(prompt).input_ids

    def preprocess_parallel_data(examples):
        results = {}
        for k, v in examples.items():
            results[k] = tokenizer(v, truncation=True, max_length=data_args.max_seq_length, add_special_tokens=False).input_ids
        return results
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_dataset = dataset.map(preprocess_parallel_data, 
                                        batched=True)

    dataset = tokenized_dataset['train'].add_column("prompt", [prompt_ids] * len(tokenized_dataset['train']))
    return dataset

def preprocess_sft_langbridge(tokenizer_llm, tokenizer_slm, data_args, training_args):
    dataset = load_dataset("json", data_files=data_args.dataset_name)
    prompt = "<|im_end|>\n<|im_start|>assistant\n"
    prompt_ids = tokenizer_llm(prompt).input_ids
    prefix = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
    prefix_ids = tokenizer_llm(prefix).input_ids

    def preprocess_parallel_data(examples):
        results = {}
        for k, v in examples.items():
            if k == 'query': tokenizer = tokenizer_slm
            elif k == 'response': tokenizer = tokenizer_llm
            else:
                continue
            results[k] = tokenizer(v, truncation=True, max_length=data_args.max_seq_length, add_special_tokens=False).input_ids
        return results
    
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_dataset = dataset.map(preprocess_parallel_data, 
                                        batched=True)
    
    dataset = tokenized_dataset['train'].add_column("prompt", [prompt_ids] * len(tokenized_dataset['train']))
    dataset = dataset.add_column("prefix", [prefix_ids] * len(dataset))
    return dataset
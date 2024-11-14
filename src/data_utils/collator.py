import torch

def parallel_data_collator(tokenizer, model_args, examples):
    src_ids = [k['src'] for k in examples]
    tgt_ids = [k['tgt'] + [tokenizer.eos_token_id] for k in examples]
    prompt_ids = [k['prompt'] for k in examples]

    src_inputs = tokenizer.pad({"input_ids": src_ids}, return_tensors="pt", padding=True)
    tgt_inputs = tokenizer.pad({"input_ids": tgt_ids}, return_tensors="pt", padding=True)
    prompt_inputs = tokenizer.pad({"input_ids": prompt_ids}, return_tensors="pt", padding=True)

    if model_args.stage == 0:
        inputs = {
            "input_ids_prompt": prompt_inputs['input_ids'],
            "attention_mask_prompt": prompt_inputs['attention_mask'],
            "input_ids_xx": src_inputs['input_ids'],
            "attention_mask_xx": src_inputs['attention_mask'],
            "input_ids_en": tgt_inputs['input_ids'],
            "attention_mask_en": tgt_inputs['attention_mask']
        }
    elif model_args.stage == 1:
        inputs = {
            "input_ids_prompt": prompt_inputs['input_ids'],
            "attention_mask_prompt": prompt_inputs['attention_mask'],
            "input_ids_en": src_inputs['input_ids'],
            "attention_mask_en": src_inputs['attention_mask'],
            "input_ids_yy": tgt_inputs['input_ids'],
            "attention_mask_yy": tgt_inputs['attention_mask']
        }
    else:
        inputs = {
            "input_ids_prompt": prompt_inputs['input_ids'],
            "attention_mask_prompt": prompt_inputs['attention_mask'],
            "input_ids_xx": src_inputs['input_ids'],
            "attention_mask_xx": src_inputs['attention_mask'],
            "input_ids_yy": tgt_inputs['input_ids'],
            "attention_mask_yy": tgt_inputs['attention_mask']
        }

    return inputs

def sft_langbridge_data_collator(tokenizer_llm, tokenizer_slm, model_args, examples):
    src_ids = [k['query'] for k in examples]
    tgt_ids = [k['response'] + [tokenizer_llm.eos_token_id] for k in examples]
    prompt_ids = [k['prompt'] for k in examples]
    prefix_ids = [k['prefix'] for k in examples]

    src_inputs = tokenizer_slm.pad({"input_ids": src_ids}, return_tensors="pt", padding=True)
    tgt_inputs = tokenizer_llm.pad({"input_ids": tgt_ids}, return_tensors="pt", padding=True)
    prompt_inputs = tokenizer_llm.pad({"input_ids": prompt_ids}, return_tensors="pt", padding=True)
    prefix_inputs = tokenizer_llm.pad({"input_ids": prefix_ids}, return_tensors="pt", padding=True)

    if model_args.stage == 0:
        inputs = {
            "input_ids_affix": prompt_inputs['input_ids'],
            "attention_mask_affix": prompt_inputs['attention_mask'],
            "input_ids_prefix": prefix_inputs['input_ids'],
            "attention_mask_prefix": prefix_inputs['attention_mask'],
            "input_ids_query": src_inputs['input_ids'],
            "attention_mask_query": src_inputs['attention_mask'],
            "input_ids_response": tgt_inputs['input_ids'],
            "attention_mask_response": tgt_inputs['attention_mask']
        }
    elif model_args.stage == 1:
        inputs = {
            "input_ids_prompt": prompt_inputs['input_ids'],
            "attention_mask_prompt": prompt_inputs['attention_mask'],
            "input_ids_en": src_inputs['input_ids'],
            "attention_mask_en": src_inputs['attention_mask'],
            "input_ids_yy": tgt_inputs['input_ids'],
            "attention_mask_yy": tgt_inputs['attention_mask']
        }
    else:
        inputs = {
            "input_ids_prompt": prompt_inputs['input_ids'],
            "attention_mask_prompt": prompt_inputs['attention_mask'],
            "input_ids_xx": src_inputs['input_ids'],
            "attention_mask_xx": src_inputs['attention_mask'],
            "input_ids_yy": tgt_inputs['input_ids'],
            "attention_mask_yy": tgt_inputs['attention_mask']
        }

    return inputs
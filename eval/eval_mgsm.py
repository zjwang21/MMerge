import json
import re
from pathlib import Path
from typing import Callable

import torch
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Sequence, List
import argparse
import os
import shutil
import pdb
from src.model_utils import ImplicitTransBridge, ItbConfig


def main(
    args,
    gsm8k_test_jsonl: str = "./data/mgsm",
    is_bf16: bool = True,
    save_dir: str  = None,
):
    batch_size = args.batch_size
    print(f"main start, is_bf16:{is_bf16}, batch_size:{batch_size}")
    
    model_path = args.model_path
    model, tokenizer_llm, tokenizer_slm = get_model(model_path, is_bf16=is_bf16)
    print("model loaded")

    batch_llama = get_batch_llama(model, tokenizer_llm, tokenizer_slm)

    if save_dir is None:
        save_dir = f"{model_path}/mgsm"
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if args.lang_only is None:
        langs = ['Swahili', 'English','Chinese','Bengali', 'German', 'Spanish', 'French', 'Japanese', 'Russian', 'Thai']
    else:
        langs = args.lang_only
    sources = []
    targets = []
    results = {}
    for lang in langs:
        print(f'===========we are testing in {lang}====================')
        
        if args.streategy == 'Parallel':
            if lang ==  'En_gsm8k':
                with open('./data/test_use.jsonl', "r", encoding='utf-8') as f:
                    gsm8k_datas = [json.loads(line) for line in f]
    
            else:
                with open(f'{gsm8k_test_jsonl}/mgsm_{lang}.json', "r", encoding='utf-8') as f:
                    gsm8k_datas = [json.loads(line) for line in f]
        else:
            if lang ==  'En_gsm8k':
                with open('./data/test_use.jsonl', "r", encoding='utf-8') as f:
                    gsm8k_datas = [json.loads(line) for line in f]
    
            else:
                with open(f'{gsm8k_test_jsonl}/mgsm_English.json', "r", encoding='utf-8') as f:
                    gsm8k_datas = [json.loads(line) for line in f]

        gen_datas_jsonl = Path(save_dir) / f"gen_{lang}_datas.jsonl"
        start_index = (
            len(open(gen_datas_jsonl).readlines()) if gen_datas_jsonl.exists() else 0
        )
        print(f"start_index: {start_index}")
        
        for i in tqdm(range(start_index, len(gsm8k_datas), batch_size)):
            cur_gsm8k_batch = gsm8k_datas[i : i + batch_size]
            input_str_list, output_str_list = gsm8k_batch_gen(lang, 
                [d["query"] for d in cur_gsm8k_batch], batch_llama, tokenizer_llm, tokenizer_slm
            )
            for j, (gsm8k_data, input_str, output_str) in enumerate(
                zip(cur_gsm8k_batch, input_str_list, output_str_list)
            ):
                with open(gen_datas_jsonl, "a") as f:
                    json.dump(
                        dict(
                            index=i + j,
                            gsm8k_data=gsm8k_data,
                            input_str=input_str,
                            output_str=output_str,
                        ),
                        f,
                    )
                    f.write("\n")

        # calculate acc
        with open(gen_datas_jsonl) as f:
            gen_datas = [json.loads(line) for line in f]

        correct_results = []
        wrong_results = []
        for gen in gen_datas:
            result = dict(
                **gen,
                extract_true_num=extract_last_num(gen["gsm8k_data"]["response"]),
                extract_pred_num=extract_last_num(gen["output_str"]),
                is_correct=None,
            )
            if abs(result["extract_true_num"] - result["extract_pred_num"]) < 1e-3:
                result["is_correct"] = True
                correct_results.append(result)
            else:
                result["is_correct"] = False
                wrong_results.append(result)

        print(f'=======done {lang}============')
        result = f"Accuracy={len(correct_results)}/({len(correct_results)}+{len(wrong_results)})={len(correct_results)/(len(correct_results) + len(wrong_results))}"
        print(result)
        with open(Path(save_dir) / f"{lang}_correct.json", "w", encoding='utf-8') as f:
            json.dump(correct_results, f, ensure_ascii=False, indent=4)
        with open(Path(save_dir) / f"{lang}_wrong.json", "w", encoding='utf-8') as f:
            json.dump(wrong_results, f, ensure_ascii=False, indent=4)
        num_result = float(result.split('=')[-1])
        if lang != 'En_gsm8k':
            results[lang] = num_result
        else:
            gsm8k = num_result
    average = sum(results.values()) / len(results)
    print(average)
    import csv
    with open(Path(save_dir) / f"MSGM_evaluate_results_bs{batch_size}.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Language', 'Accuracy'])
        for key, value in results.items():
            writer.writerow([key, value])
        writer.writerow(['Average', average])
        # writer.writerow(['GSM8K', gsm8k])
    


def gsm8k_batch_gen(
    lang_, gsm8k_questions, batch_llm, tokenizer_llm, tokenizer_slm
):  
    bsz = len(gsm8k_questions)
    prompt = "<|im_end|>\n<|im_start|>assistant\n"
    prompts = tokenizer_llm([prompt] * bsz, add_special_tokens=False, return_tensors='pt', padding=True)
    prefix = "<|im_start|>system\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>user\n"
    prefixs = tokenizer_llm([prefix] * bsz, add_special_tokens=False, return_tensors='pt', padding=True)
    querys = tokenizer_slm(gsm8k_questions, truncation=True, max_length=1024, add_special_tokens=False, return_tensors='pt', padding=True)
    inputs = {
            "input_ids_affix": prompts['input_ids'],
            "attention_mask_affix": prompts['attention_mask'],
            "input_ids_prefix": prefixs['input_ids'],
            "attention_mask_prefix": prefixs['attention_mask'],
            "input_ids_query": querys['input_ids'],
            "attention_mask_query": querys['attention_mask'],
        }
    output_str_list = batch_llm(inputs)
    return gsm8k_questions, output_str_list


def get_batch_llama(model, tokenizer_llm, tokenizer_slm):
    @torch.inference_mode()
    def batch_llama(inputs):
        output_ids = model(**inputs).tolist()
        # pad_token_id=tokenizer.eos_token_id
        #print(output_ids)
        real_output_ids = [
            output_id[len(inputs['input_ids_query'][i]) + len(inputs['input_ids_prefix'][i]) + len(inputs['input_ids_affix'][i]) :] for i, output_id in enumerate(output_ids)
        ]
        output_strs = tokenizer_llm.batch_decode(real_output_ids, skip_special_tokens=True)
        return output_strs

    return batch_llama


def get_model(model_path: str, is_bf16: bool = False):
    print(model_path)
    model = ImplicitTransBridge.from_pretrained(model_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(model.config.llm_path, padding_side="left")
    tokenizer_slm = AutoTokenizer.from_pretrained(model.config.slm_path_a)
    model = model.cuda().to(torch.bfloat16)
    model.eval()
    print(model.dtype)

    return model, tokenizer_llm, tokenizer_slm


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", "\g<1>\g<2>", text)  # 处理形如 123,456
    res = re.findall(r"(\d+(\.\d+)?)", text)  # 匹配 123456.789
    if len(res) > 0:
        num_str = res[-1][0]
        return float(num_str)
    else:
        return 0.0


if __name__ == "__main__":
    import fire

    parser = argparse.ArgumentParser(description="Eval the finetued SFT model")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to baseline model",
        required=True,
    )
    parser.add_argument(
        "--streategy",
        type=str,
        help="which streategy to evaluate the model",
        required=True,
        choices=['Parallel','Cross']
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batchsize",
        required=True
    )
    parser.add_argument(
        "--lang_only",
        type=str,
        nargs='+',
        help="specific language to test",
        default = None
    )
    args = parser.parse_args()

    fire.Fire(main(args=args))
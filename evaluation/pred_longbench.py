import os
import torch
import argparse
import json
import random
import transformers
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

import pdb

def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name.lower():
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name.lower():
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name.lower() or "vicuna" in model_name.lower():
        from fastchat.model import get_conversation_template
        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name.lower() or "llama-2" in model_name.lower():
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name.lower():
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name.lower():
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif ("llama3" in model_name.lower() or "llama-3" in model_name.lower()) and "instruct" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    elif "mistral-7b-instruct-v0.3" in model_name.lower():
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    elif "qwen" in model_name.lower():
        messages = [
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = True)
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response

def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
):
    preds = []
    for json_obj in tqdm(data):
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(prompt, truncation = False, return_tensors = "pt").input_ids[0]
        # pdb.set_trace()
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(tokenized_prompt[ : half], skip_special_tokens = True) + tokenizer.decode(tokenized_prompt[-half : ], skip_special_tokens = True)
        if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
            prompt = build_chat(tokenizer, prompt, model_name)
        input = tokenizer(prompt, truncation = False, return_tensors = "pt").to(device)
        context_length = input.input_ids.shape[-1]
        # pdb.set_trace()
        if dataset == "samsum":
            output = model.generate(
                **input,
                max_new_tokens = max_gen,
                num_beams = 1,
                do_sample = False,
                temperature = 1.0,
                # past_key_values = BfpDynamicCache(),
                min_length = context_length + 1,
                eos_token_id=[tokenizer.eos_token_id, tokenizer.encode("\n", add_special_tokens=False)[-1]],
            )[0]
        else:
            # try:
            # model.config.use_cache = True
            output = model.generate(
                **input,
                max_new_tokens = max_gen,
                num_beams = 1,
                do_sample = False,
                temperature = 1.0,
                # past_key_values = BfpDynamicCache(),
            )[0]
        pred = tokenizer.decode(output[context_length : ], skip_special_tokens = True)
        pred = post_process(pred, model_name)
        preds.append({"pred": pred, "answers": json_obj["answers"], "all_classes": json_obj["all_classes"], "length": json_obj["length"]})
    return preds

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def longbenchpred(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    args: argparse.Namespace,
):
    seed_everything(42)
    
    model2path = json.load(open("./longbenchconfig/model2path.json", "r"))
    model2maxlen = json.load(open("./longbenchconfig/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = args.model_name_or_path.split("/")[-1]
    model.to(device)
    model.eval()
    max_length = model2maxlen[model_name]
    if args.longbench_e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", 
                    "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # datasets = ["2wikimqa", "repobench-p", "hotpotqa", "trec", "multi_news", "multifieldqa_en", "qasper", "qmsum", "triviaqa"]
        datasets = ["multifieldqa_en"]

    # keep the same prompt as the original longbench
    dataset2prompt = json.load(open("./longbenchconfig/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("./longbenchconfig/dataset2maxlen.json", "r"))
    os.makedirs("./longbench_pred", exist_ok= True)
    os.makedirs("./longbench_pred_e", exist_ok= True)
    for dataset in datasets:
        if args.longbench_e:
            data = load_dataset('THUDM/LongBench', f"{dataset}_e", split = 'test')
            os.makedirs(f"./longbench_pred_e/{model_name}", exist_ok= True)
            pred_out_path = f"./longbench_pred_e/{model_name}/"
            out_path = f"./longbench_pred_e/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset('THUDM/LongBench', dataset, split = 'test')
            os.makedirs(f"./longbench_pred/{model_name}", exist_ok= True)
            pred_out_path = f"./longbench_pred/{model_name}/"
            out_path = f"./longbench_pred/{model_name}/{dataset}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(
            model = model,
            tokenizer = tokenizer,
            data = data,
            max_length = max_length,
            max_gen = max_gen,
            prompt_format = prompt_format,
            dataset = dataset,
            device = device,
            model_name = model_name
        )
        with open(out_path, "w", encoding = "utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii= False)
                f.write("\n")
    
    return pred_out_path
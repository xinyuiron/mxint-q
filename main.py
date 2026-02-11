import os
import torch
import argparse
from transformers import AutoTokenizer, AutoConfig
from pathlib import Path

from utils.argutils import setup_parser
from utils.loggerutils import create_logger
from evaluation.pred_longbench import longbenchpred
from evaluation.eval_longbench import longbencheval
from quant.mxintquant import mxintquant

import pdb

def main(args):
    model_name_or_path = args.model_name_or_path
    model_name = model_name_or_path.split("/")[-1]

    if args.log_dir:
        args.log_dir = os.path.join(args.log_dir, model_name)
        Path(args.log_dir).mkdir(parents = True, exist_ok = True)
    
    log_dir = Path(args.log_dir)
    logger = create_logger(log_dir, name = "mxint-q")
    logger.info(args)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast = False, trust_remote_code = True)

    # load config
    config = AutoConfig.from_pretrained(model_name_or_path)
    config._attn_implementation = "eager"

    if "llama" in model_name_or_path.lower():
        from models.llama_mxint import LlamaForCausalLM_MXINTQ
        model = LlamaForCausalLM_MXINTQ.from_pretrained(
            pretrained_model_name_or_path = model_name_or_path,
            config = config,
            low_cpu_mem_usage = True,
            torch_dtype = torch.float16,
            device_map = "cpu",
        )
    elif "mistral" in model_name_or_path.lower():
        from models.mistral_mxint import MistralForCausalLM_MXINTQ
        model = MistralForCausalLM_MXINTQ.from_pretrained(
            pretrained_model_name_or_path = model_name_or_path,
            config = config,
            low_cpu_mem_usage = True,
            torch_dtype = torch.float16,
            device_map = "cpu",
        )
    else:
        raise NotImplementedError(f"Model {model_name_or_path} is not supported!")

    args.mxint_quant_params = {
        "weight_quant_params": {
            "n_bits": 4,
            "group_size": 128,
            "dimension": "per_channel",
            "symmetric": False,
        },
        "act_quant_params": {
            "n_bits": 9,
            "group_size": 32,
            "dimension": "per_token"
        },
        "q_quant_params":
        {
            "n_bits": 9,
            "group_size": 32,
            "dimension": "per_token"
        },
        "k_quant_params":
        {
            "n_bits": 9,
            "group_size": 32,
            "dimension": "per_token",
        },
        "v_quant_params":
        {
            "n_bits": 9,
            "group_size": 32,
            "dimension": "per_channel",
        },
        "attnw_quant_params":
        {
            "n_bits": 9,
            "group_size": 32,
            "dimension": "per_token",
        }
    }

    # pdb.set_trace()

    mxintquant(model, args, logger)

    pred_out_path = longbenchpred(
        model = model,
        tokenizer = tokenizer,
        args = args
    )

    longbencheval(
        path = pred_out_path,
        args = args,
        logger = logger
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Arguments Parser", parents = [setup_parser()])
    args = parser.parse_args()
    main(args)
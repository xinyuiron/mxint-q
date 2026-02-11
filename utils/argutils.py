import argparse

# MODEL_LIST = [
#     "meta-llama/Llama-3.2-1B-Instruct",
#     "meta-llama/Llama-3.2-3B-Instruct",
#     "meta-llama/Llama-3.1-8B-Instruct",   
#     "mistralai/Mistral-7B-Instruct-v0.3",
#     "meta-llama/Llama-2-13b-chat-hf",
# ]

def setup_parser():
    parser = argparse.ArgumentParser("Arguments Parser", add_help = False)

    # General arguments
    parser.add_argument("--model_name_or_path", "-m", type = str, default = "meta-llama/Llama-3.2-3B-Instruct", help = "Model name or path.")
    parser.add_argument("--log_dir", type = str, default = "./logs", help = "Directory of logging file.")

    # weight quant arguments
    # parser.add_argument("--wquant", action = "store_true", help = "Whether to use weight quantization.")
    # parser.add_argument("--wbits", type = int, default = 4, help = "Weight bits for quantization.")
    # parser.add_argument("--abits", type = int, default = 9, help = "Activation bits for quantization.")
    # parser.add_argument("--w_group_size", type = int, default = 128, help = "Weight group size for quantization.")
    # parser.add_argument("--a_group_size", type = int, default = 32, help = "Activation group size for quantization.")
    # parser.add_argument("--deactive_amp", action = "store_true", help="Deactivate AMP when 8<=bits<16")
    # parser.add_argument("--aug_loss", type = bool, default = False, help = "Whether to calculate additional loss with same input")
    # parser.add_argument("--epochs", type = int, default = 0, help = "Number of epochs for training. Default is 0 (no training).")
    # parser.add_argument("--calibration_dataset", type = str, default = "wikitext2", help = "Dataset for calibration. Default is 'wikitext2'.")
    # parser.add_argument("--calibration_samples", type = int, default = 128, help = "Number of samples for calibration. Default is 128.")
    # parser.add_argument("--calibration_batch_size", type = int, default = 1, help = "Batch size for calibration. Default is 1.")
    # parser.add_argument("--calibration_seed", type = int, default = 2, help = "Random seed for calibration. Default is 2.")
    # parser.add_argument("--w_resume", type = str, default = None, help = "Path to the pre-trained parameters for weight-only quantization.")
    # parser.add_argument("--weight_decay", type = float, default = 0.0, help = "Weight decay for optimizer. Default is 0.0.")
    # parser.add_argument("--lwc", action = "store_true", default = False, help = "Whether to use learnable weight clipping (LWC).")
    # parser.add_argument("--lwc_lr", type = float, default = 1e-2, help = "Learning rate for learnable weight clipping (LWC). Default is 1e-2.")
 
    parser.add_argument("--longbench", action = "store_true", help = "Whether to use LongBench for evaluation.")
    parser.add_argument("--longbench_e", action = "store_true", help = "Whether to evaluate on LongBench-E.")
    
    return parser
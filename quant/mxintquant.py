import torch
from quant.intlinear import QuantLinear

def mxintquant(model, args, logger):
    for param in model.parameters():
        param.requires_grad = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = model.model.layers

    for i in range(len(layers)):
        logger.info(f"=== Start quantizing layer {i}'s weight ===")
        layer = layers[i].to(device)
        layer.turn_to_quantmode(args.mxint_quant_params)

        for _, module in layer.named_modules():
            if isinstance(module, QuantLinear):
                module.weight = module.weight_quantizer(module.weight)
        
        layers[i] = layer.to("cpu")
        del layer

    return model

from torch.nn import Module


def get_named_modules_mapping(module: Module):
    rv = {}
    for name, submodule in module.named_modules():
        rv[name] = submodule
    return rv


def count_module_parameters(module: Module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

import torchvision.models as models

# Model under test
rn18 = models.resnet18()

from torch.fx.passes.shape_prop import ShapeProp

from torchinspect.module import count_module_parameters, get_named_modules_mapping
from torchinspect.tracer import ModuleNodeTracer

# Instantiate our ModulePathTracer and use that to trace our ResNet18
tracer = ModuleNodeTracer()
traced_rn18 = tracer.trace(rn18)

shape_int = ShapeProp(traced_rn18)

# Print (node, module qualified name) for every node in the Graph

named_modules = get_named_modules_mapping(rn18)

for node in traced_rn18.nodes:
    module_name = tracer.get_module_name_by_node(node)
    module = named_modules[module_name]
    print(module_name, '->', count_module_parameters(module))

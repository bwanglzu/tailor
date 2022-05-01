from typing import Optional

from torch import Tensor
from torch.fx import GraphModule, Tracer
from torch.fx.passes.shape_prop import ShapeProp as ShapeInterpreter
from torch.nn.modules import Module

from ._symbolic_trace import symbolic_trace_with_custom_tracer
from .module import count_module_parameters, get_named_modules_mapping
from .tracer import ModuleNodeTracer
from .visualizer import VisualizerMixin


class Interpreter(VisualizerMixin):
    def __init__(self, tracer: Optional[Tracer] = None):
        super().__init__()
        self.tracer = tracer if tracer else ModuleNodeTracer()
        self.visualizer = VisualizerMixin()

    def interpret(self, module: Module, input_: Tensor):
        rv = []
        name_module_mapping = get_named_modules_mapping(module)
        traced: GraphModule = symbolic_trace_with_custom_tracer(
            module, tracer=self.tracer
        )
        self.shape_interpreter = ShapeInterpreter(traced)
        self.shape_interpreter.propagate(input_)
        for node in traced.graph.nodes:
            module_name = self.tracer.get_module_name_by_node(node)
            if module_name and name_module_mapping.get(module_name):
                item = {}
                current_module = name_module_mapping[module_name]
                num_params = count_module_parameters(current_module)
                trainable = True if num_params > 0 else False
                item['name'] = node.name
                item['dtype'] = node.meta['tensor_meta'].dtype
                item['shape'] = list(node.meta['tensor_meta'].shape)
                item['num_params'] = num_params
                item['trainable'] = trainable
                rv.append(item)
        self.visualizer.plot(module=module, summaries=rv)

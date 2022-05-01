from typing import Any, Callable, Dict, Optional, Union

import torch
from torch.fx import GraphModule, Tracer


def symbolic_trace_with_custom_tracer(
    root: Union[torch.nn.Module, Callable[..., Any]],
    concrete_args: Optional[Dict[str, Any]] = None,
    tracer: Optional[Tracer] = None,
) -> GraphModule:
    if tracer:
        custom_tracer = tracer
    else:
        custom_tracer = Tracer()
    graph = custom_tracer.trace(root, concrete_args)
    name = (
        root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    )
    return GraphModule(custom_tracer.root, graph, name)

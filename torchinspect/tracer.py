from typing import Any, Callable, Dict, Optional, Tuple, Union

from torch.fx import Node, Proxy, Tracer
from torch.fx.node import Target
from torch.nn import Module


class ModuleNodeTracer(Tracer):

    current_module_qualified_name: str = ''
    node_to_originating_module: Dict[Node, str] = {}
    originating_module_to_node: Dict[str, Node] = {}

    def call_module(
        self,
        module: Module,
        forward: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        prev_qualified_name = self.current_module_qualified_name
        try:
            self.current_module_qualified_name = self.path_of_module(module)
            return super().call_module(module, forward, args, kwargs)
        finally:
            self.current_module_qualified_name = prev_qualified_name

    def create_proxy(
        self,
        kind: str,
        target: Target,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
        proxy_factory_fn: Callable[[Node], 'Proxy'] = None,
    ):
        proxy = super().create_proxy(
            kind, target, args, kwargs, name, type_expr, proxy_factory_fn
        )
        self.node_to_originating_module[proxy.node] = self.current_module_qualified_name
        self.originating_module_to_node[self.current_module_qualified_name] = proxy.node
        return proxy

    def create_node(
        self,
        kind: str,
        target: Union[str, Callable],
        args: Tuple[Any],
        kwargs: Dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> Node:
        # if kind == 'call_module':
        node = super().create_node(kind, target, args, kwargs, name)
        return node

    def get_module_name_by_node(self, node: Node):
        return self.node_to_originating_module.get(node, None)

    def get_node_by_module_name(self, module_name: str):
        return self.originating_module_to_node.get(module_name, None)

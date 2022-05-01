from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch.fx.proxy
from torch.fx import Node, Proxy, Tracer
from torch.fx.node import Target
from torch.nn import Module, Sequential


class ModuleNodeTracer(Tracer):

    current_module_qualified_name: str = ''
    node_to_originating_module: Dict[Node, str] = {}
    originating_module_to_node: Dict[str, Node] = {}

    def __init__(self, leaf_module: Optional[Module] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._leaf_module = leaf_module

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
        except torch.fx.proxy.TraceError:
            # annotate module with control flow as leaf node.
            module._is_leaf_module = True
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
        node = super().create_node(kind, target, args, kwargs, name)
        return node

    def get_module_name_by_node(self, node: Node):
        return self.node_to_originating_module.get(node, None)

    def get_node_by_module_name(self, module_name: str):
        return self.originating_module_to_node.get(module_name, None)

    def is_leaf_module(self, m: Module, module_qualified_name: str) -> bool:
        if self._leaf_module and isinstance(m, self._leaf_module):
            return True

        if hasattr(m, '_is_leaf_module') and m._is_leaf_module:
            return True

        return m.__module__.startswith('torch.nn') and not isinstance(m, Sequential)

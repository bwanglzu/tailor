from typing import Optional

import torch
from torch.fx import GraphModule


class SublayerRewriter(object):
    def delete(self, layer: str) -> GraphModule:
        """Delete a specific layer by name."""
        traced = self.graph_module
        for node in traced.graph.nodes:
            module_name = self._tracer.get_module_name_by_node(node)
            if layer == module_name:
                traced.delete_submodule(layer)
                with traced.graph.inserting_after(node):
                    node.replace_all_uses_with(node.prev)
                traced.graph.erase_node(node)
                break
        return traced

    def insert(self, module: torch.nn.Module, name: str, at: Optional[str] = None):
        """Insert a layer after a specific layer

        :param module: The `torch.nn.Module` object to be inserted.
        :param name: The name of the insert submodule.
        :param at: The name of the layer to insert after, by default insert at the
          end of the model.
        """
        traced = self.graph_module
        if not at:
            at = self._interpret()[-1]['name']
        for node in traced.graph.nodes:
            module_name = self._tracer.get_module_name_by_node(node)
            if at == module_name:
                traced.add_submodule(name, module)
                with traced.graph.inserting_after(node):
                    new_node = traced.graph.call_module(name, node.args, node.kwargs)
                    node.replace_all_uses_with(new_node)
                break
        return traced

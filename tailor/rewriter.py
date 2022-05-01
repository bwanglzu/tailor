import torch
from torch.fx import GraphModule


class SublayerRewriter(object):
    def remove(self, layer: str) -> tuple[GraphModule, bool]:
        """Remove a specific layer by name."""
        traced = self.trace()
        prev_node = None
        for node in traced.graph.nodes:
            module_name = self._tracer.get_module_name_by_node(node)
            if layer == module_name:
                traced.delete_submodule(layer)
                with traced.graph.inserting_after(node):
                    new_node = prev_node
                    node.replace_all_uses_with(new_node)
                traced.graph.erase_node(node)
            prev_node = node
        traced.recompile()
        return traced

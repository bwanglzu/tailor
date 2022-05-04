from torch.fx import GraphModule

from .tracer import ModuleNodeTracer


class SublayerRewriter(object):
    def delete(self, layer: str) -> GraphModule:
        """Delete a specific layer by name."""
        prev_node = None
        traced = self.graph_module
        for node in traced.graph.nodes:
            module_name = self._tracer.get_module_name_by_node(node)
            if layer == module_name:
                traced.delete_submodule(layer)
                with traced.graph.inserting_after(node):
                    new_node = prev_node
                    node.replace_all_uses_with(new_node)
                traced.graph.erase_node(node)
                break
            prev_node = node
        return traced

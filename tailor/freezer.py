from typing import Optional

from .utils import get_named_modules_mapping


class LayerNotFoundError(Exception):
    ...


class FreezerMixin(object):
    def freeze(self, from_: Optional[str] = None, to: Optional[str] = None):
        all_names = list(get_named_modules_mapping(self._model).keys())
        if from_ and from_ not in all_names:
            msg = (
                f'The layer {from_} not exist in all layers.'
                'Please call `plot()` to see the available layers.'
            )
            raise LayerNotFoundError(msg)
        if to and to not in all_names:
            msg = (
                f'The layer {to} not exist in all layers.'
                'Please call `plot()` to see the available layers.'
            )
            raise LayerNotFoundError(msg)
        if from_:
            requires_grad = True
        else:
            requires_grad = False
        for name, param in self._model.named_parameters():
            if from_ and from_ in name:
                requires_grad = False
            if to and to in name:
                requires_grad = True
            param.requires_grad = requires_grad

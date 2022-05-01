from typing import Union

import torch
from rich.console import Console
from rich.table import Table


class VisualizerMixin(object):
    def __init__(self):
        self._console = Console()
        self._table = Table()
        self._init_table()

    def _init_table(self):
        self._table.add_column('name', justify='right', style='cyan1', no_wrap=True)
        self._table.add_column('dtype', style='magenta', justify='right', no_wrap=True)
        self._table.add_column(
            'num_params', justify='right', style='light_sea_green', no_wrap=True
        )
        self._table.add_column(
            'shape', justify='right', style='deep_sky_blue2', no_wrap=True
        )
        self._table.add_column(
            'trainable', justify='right', style='green3', no_wrap=True
        )

    def plot(self, module: torch.nn.Module, input_shape: Union[list, tuple]):
        self._table_name = f'Model Structure: {module.__class__.__name__}'
        self._table.title = self._table_name
        for summary in self._interpret(module=module, input_shape=input_shape):
            self._table.add_row(
                summary['name'],
                str(summary['dtype']),
                str(summary['num_params']),
                str(summary['shape']),
                str(summary['trainable']),
            )
        self._console.print(self._table)

from rich.console import Console
from rich.table import Table


class VisualizerMixin(object):
    def __init__(self):
        self._console = Console()
        self._table = Table()
        self._init_table()

    def _init_table(self):
        self._table.add_column('name', justify='left', style='blue', no_wrap=True)
        self._table.add_column('dtype', style='magenta', justify='right', no_wrap=True)
        self._table.add_column(
            'num_params', justify='right', style='light_sea_green', no_wrap=True
        )
        self._table.add_column(
            'shape', justify='right', style='deep_sky_blue2', no_wrap=True
        )
        self._table.add_column(
            'trainable', justify='right', style='green4', no_wrap=True
        )

    def plot(self):
        self._table_name = f'Model Structure: {self._model.__class__.__name__}'
        self._table.title = self._table_name
        for summary in self._interpret():
            self._table.add_row(
                summary['name'],
                str(summary['dtype']),
                str(summary['num_params']),
                str(summary['shape']),
                str(summary['trainable']),
            )
        self._console.print(self._table)

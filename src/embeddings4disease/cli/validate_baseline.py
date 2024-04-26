import os
import typing as tp
import yaml

import click

from embeddings4disease.metrics import metrics


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path: str) -> None:
    with open(os.path.abspath(config_path)) as f:
        config: dict[str, tp.Any] = yaml.safe_load(f)

    baseline = metrics.Baseline(config)

    print(baseline.get_metrics_value())

if __name__ == "__main__":
    validate()

# import warnings

import click

from factories import abstract_factory


# warnings.filterwarnings("ignore")


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def validation(config_path: str) -> None:
    factory = abstract_factory.AbstractFactory().create(config_path)
    model = factory.create_model()

    metric_computer, used_metrics = factory.create_metric_computer()

    print(metric_computer.get_metrics_value(model, **used_metrics))


if __name__ == "__main__":
    validation()

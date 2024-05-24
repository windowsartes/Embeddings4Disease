import click

from embeddings4disease.head.factories import abstract_factory


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path: str) -> None:
    factory = abstract_factory.AbstractFactory().create(config_path)
    model = factory.create_model()

    metric_computer = factory.create_metric_computer()

    print(metric_computer.get_metrics_value(model))


if __name__ == "__main__":
    validate()

import click

from data.preprocessing import abstract_factory


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def preprocessing(config_path: str) -> None:
    factory = abstract_factory.AbstractFactory().create(config_path)
    factory.train_val_split()
    factory.create_vocab()


if __name__ == "__main__":
    preprocessing()

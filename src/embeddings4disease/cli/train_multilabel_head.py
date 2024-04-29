import click

from embeddings4disease.head.factories import abstract_factory
from embeddings4disease.trainer import Trainer


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path: str) -> None:
    factory = abstract_factory.AbstractFactory().create(config_path)

    factory.initialize()

    model = factory.create_model()
    training_args = factory.create_training_args()

    dataloader_train = factory.create_dataloader("training")
    dataloader_eval = factory.create_dataloader("validation")

    callbacks = factory.create_callbacks()

    trainer = Trainer(
        model=model,
        train_dataloader=dataloader_train,
        eval_dataloader=dataloader_eval,
        callbacks=callbacks,
        args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    train()

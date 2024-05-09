import click

from embeddings4disease.head.factories import abstract_factory
from embeddings4disease.trainer.trainer import Trainer


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path: str) -> None:
    factory = abstract_factory.AbstractFactory().create(config_path)

    factory.initialize()

    model = factory.create_model()
    training_args = factory.create_training_args()

    data_collator = factory.create_collator()
    dataset_train = factory.create_dataset("training")
    dataset_eval = factory.create_dataset("validation")

    callbacks = factory.create_callbacks()

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        callbacks=callbacks,
        args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    train()

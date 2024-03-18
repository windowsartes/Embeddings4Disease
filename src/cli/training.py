import warnings

import click
import transformers

from factories import abstract_factory


warnings.filterwarnings("ignore")


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path: str) -> None:
    factory = abstract_factory.AbstractFactory().create(config_path)

    model = factory.create_model()
    training_args = factory.create_training_args()

    data_collator = factory.create_collator()
    dataset_train = factory.create_dataset("training")
    dataset_eval = factory.create_dataset("validation")

    callbacks = factory.create_callbacks()

    factory.set_warmup_epochs(training_args, dataset_train)

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        callbacks=callbacks,
    )

    trainer.train()


if __name__ == "__main__":
    train()

import os
import pathlib
import warnings

import click
import transformers

from factories import abstract_factory


warnings.filterwarnings("ignore")

working_dir = pathlib.Path(os.getcwd())


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def training(config_path: str) -> None:
    factory = abstract_factory.AbstractFactory().create(config_path)
    factory.create_storage()

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

    factory.optionally_save(trainer)
    #trainer.save_model(working_dir.joinpath("saved_model"))


if __name__ == "__main__":
    training()

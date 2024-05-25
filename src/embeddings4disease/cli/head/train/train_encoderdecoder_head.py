import warnings

import click
import transformers

from embeddings4disease.head.factories import abstract_factory


warnings.simplefilter(action="ignore", category=FutureWarning)

@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path: str) -> None:
    factory = abstract_factory.AbstractFactory().create(config_path)

    factory.initialize()

    tokenizer = factory.load_tokenizer()

    model = factory.create_model()

    training_args = factory.create_training_args()

    data_collator = factory.create_collator()
    dataset_train = factory.create_dataset("training")
    dataset_eval = factory.create_dataset("validation")

    factory.set_warmup_epochs(training_args, dataset_train)

    metric_computer = factory.create_metric_computer()

    callbacks = factory.create_callbacks()

    trainer = transformers.Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=metric_computer,
        callbacks=callbacks,
    )

    trainer.train()


if __name__ == "__main__":
    train()

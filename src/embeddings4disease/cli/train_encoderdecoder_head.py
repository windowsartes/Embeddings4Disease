import click
import transformers

from embeddings4disease.head.factories import abstract_factory


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

    callbacks = factory.create_callbacks()

    trainer = transformers.Seq2SeqTrainer(
        model,
        training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=callbacks,
        # compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    train()

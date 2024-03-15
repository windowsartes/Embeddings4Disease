import click
import transformers


from factories import abstract_factory


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path: str):

    factory = abstract_factory.AbstractFactory().create(config_path)
    model, training_args, data_collator, dataset_train, dataset_eval, callbacks = (
        factory.construct()
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_train,
        eval_dataset=dataset_eval,
        callbacks=callbacks,
    )

    print(model)
    print(callbacks)

    # trainer.train()


if __name__ == "__main__":
    train()

import os
import pathlib
import typing as tp
import yaml

from embeddings4disease.factories import architecture_factory


working_dir: pathlib.Path = pathlib.Path(os.getcwd())


class AbstractFactory:
    @staticmethod
    def create(config_path: str) -> architecture_factory.ArchitectureFactory:
        with open(working_dir.joinpath(config_path)) as f:
            config: dict[str, tp.Any] = yaml.safe_load(f)

        return architecture_factory.CLASS_REGISTER[config["model"]["type"]](config)

import os
import pathlib
import typing as tp
import yaml

from embeddings4disease.data.preprocessing import preprocessor_factory


working_dir: pathlib.Path = pathlib.Path(os.getcwd())


class AbstractFactory:
    @staticmethod
    def create(config_path: str) -> preprocessor_factory.PreprocessorFactory:
        with open(working_dir.joinpath(config_path)) as f:
            config: dict[str, tp.Any] = yaml.safe_load(f)

        return preprocessor_factory.CLASS_REGISTER[config["dataset"]](config)

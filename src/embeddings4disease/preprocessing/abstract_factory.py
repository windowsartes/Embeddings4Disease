import os
import typing as tp
import yaml

from embeddings4disease.preprocessing import preprocessor_factory


class AbstractFactory:
    @staticmethod
    def create(config_path: str) -> preprocessor_factory.PreprocessorFactory:
        with open(os.path.abspath(config_path)) as f:
            config: dict[str, tp.Any] = yaml.safe_load(f)

        return preprocessor_factory.PREPROCESSOR_REGISTER[config["dataset"]](config)

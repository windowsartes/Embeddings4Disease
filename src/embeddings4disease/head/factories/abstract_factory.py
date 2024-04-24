import os
import typing as tp
import yaml

from embeddings4disease.head.factories import architecture_factory


class AbstractFactory:
    @staticmethod
    def create(config_path: str) -> architecture_factory.HeadFactory:
        with open(os.path.abspath(config_path)) as f:
            config: dict[str, tp.Any] = yaml.safe_load(f)

        return architecture_factory.CLASS_REGISTER[config["model"]["type"]](config)

import os
import typing as tp
import yaml

from embeddings4disease.head.factories import head_factory


class AbstractFactory:
    @staticmethod
    def create(config_path: str) -> head_factory.HeadFactory:
        with open(os.path.abspath(config_path)) as f:
            config: dict[str, tp.Any] = yaml.safe_load(f)

        return head_factory.HEAD_REGISTER[config["model"]["type"]](config)

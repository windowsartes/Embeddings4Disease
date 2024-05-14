import os
import typing as tp
import yaml

from embeddings4disease.backbone.factories import backbone_factory


class AbstractFactory:
    @staticmethod
    def create(config_path: str) -> backbone_factory.BackboneFactory:
        with open(os.path.abspath(config_path)) as f:
            config: dict[str, tp.Any] = yaml.safe_load(f)

        return backbone_factory.BACKBONE_REGISTER[config["model"]["type"]](config)

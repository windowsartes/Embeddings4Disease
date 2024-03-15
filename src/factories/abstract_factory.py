import os
import pathlib
import yaml

from factories import architecture_factory


# dir_path: str = os.path.dirname(os.path.realpath(__file__))
# root_path: pathlib.Path = pathlib.Path(dir_path).parents[1]

working_dir = pathlib.Path(os.getcwd())


class AbstractFactory:
    @staticmethod
    def create(config_path: str) -> architecture_factory.C:
        config_path = working_dir.joinpath(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)
            
        return architecture_factory.NAME_TO_CLASS[config["model"]["type"]](config)

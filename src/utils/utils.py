import pathlib


def create_dir(path: pathlib.Path):
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    
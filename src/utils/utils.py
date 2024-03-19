import os
import pathlib
import shutil


def create_dir(path: str | pathlib.Path) -> None:
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def delete_files(path: str | pathlib.Path) -> None:
    for filename in os.listdir(str(path)):
        file_path = pathlib.Path(path).joinpath(filename)

        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

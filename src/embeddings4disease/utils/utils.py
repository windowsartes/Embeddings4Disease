import importlib.resources
import os
import pathlib
import platform
import shutil
import subprocess
import typing as tp
from datetime import datetime

import pandas as pd


class CodeCutter:
    """
    It's a simple functor. its only task is to cut given codes to the required number of characters.

    Args:
        code_length (int, optional): number of first characters to leave. Defaults to 3.
    """
    def __init__(self, code_length: int = 3) -> None:
        self.code_length: int = code_length

    def __call__(self, code: str) -> str:
        return code[:self.code_length]


class CustomOrderedSet:
    """
    It's just a simle ordered set, based on python list.
    """
    def __init__(self) -> None:
        self.storage: list[tp.Any] = []

    def add(self, value: tp.Any) -> None:
        if value not in self.storage:
            self.storage.append(value)

    def __contains__(self, value: tp.Any) -> bool:
        return value in self.storage

    def __iter__(self):  # type: ignore
        self.current_index: int = 0

        return self

    def __next__(self) -> tp.Any:
        if self.current_index < len(self.storage):
            x: tp.Any = self.storage[self.current_index]
            self.current_index += 1

            return x

        raise StopIteration


def create_dir(path: str | pathlib.Path) -> None:
    """
    Creates a directory by given path.

    Args:
        path (str | pathlib.Path): path to the directory.
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)

def delete_files(path: str | pathlib.Path) -> None:
    """
    Deletes all the files from directory by given path.

    Args:
        path (str | pathlib.Path): path to the directory.
    """
    for filename in os.listdir(str(path)):
        file_path = pathlib.Path(path).joinpath(filename)

        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def get_project_root() -> pathlib.Path:
    """
    Returns a path to the package base directory.

    Returns:
        pathlib.Path: this package directory inside your system.
    """
    with importlib.resources.path("embeddings4disease", "__init__.py") as src_path:
        path = src_path.parents[2]
    return path

def get_cwd() -> str:
    """
    Returns your current working directory.

    Returns:
        str: your cwd.
    """
    if platform.system() == "Windows":
        result = subprocess.run("cd", text=True, stdout=subprocess.PIPE, shell=True)
        return result.stdout.strip()

    result = subprocess.run("pwd", text=True, stdout=subprocess.PIPE, shell=True)
    return result.stdout.strip()

def to_datetime(x: pd.Timestamp) -> datetime:
    """
    Cast pandas.Timestamp into datetime

    Args:
        x (pd.Timestamp): date as a timestamp.

    Returns:
        datetime: date as a datetime.
    """
    return x.to_pydatetime()

def str2datetime(column: pd.Series) -> pd.Series:
    """
    Used to cast string date representation into datatime

    Args:
        column (pd.Series): colomn containig dates as string.

    Returns:
        pd.Series: pandas.Series containing dates as datetime.
    """
    return pd.to_datetime(column).apply(to_datetime)

def get_year(x: pd.Timestamp) -> int:
    """
    Used to get year value from pandas Timestemp.

    Args:
        x (pd.Timestamp): timestamp

    Returns:
        int: year's value
    """
    return x.year

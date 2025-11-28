from json import dump as json_dump
from typing import TYPE_CHECKING
from os import makedirs as os_makedirs

from cv2 import imwrite as cv2_imwrite

if TYPE_CHECKING:
    from pathlib import Path

    from numpy import ndarray


def read_text(path: "Path") -> str | None:
    if not path.exists():
        return None

    with open(path, "r", encoding="utf8") as file:
        return file.read()

    return None


def write_text(path: "Path", text: str) -> bool:
    with open(path, 'w', encoding="utf8") as file:
        file.write(text)
        return True

    return False


def write_json(path: "Path", data: dict, indent: int = 2) -> bool:
    with open(path, 'w') as file:
        json_dump(data, file, indent=indent)
        return True

    return False


def write_image(path: "Path", image: "ndarray") -> bool:
    cv2_imwrite(path, image)


def makedirs(path: "Path") -> bool:
    os_makedirs(path, ok_exist=True)

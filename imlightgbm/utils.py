import logging
from typing import Callable


def _modify_docstring(docstring: str) -> str:
    lines = docstring.splitlines()

    feval_start = next(i for i, line in enumerate(lines) if "feval" in line)
    init_model_start = next(i for i, line in enumerate(lines) if "init_model" in line)
    del lines[feval_start:init_model_start]

    note_start = next(i for i, line in enumerate(lines) if "Note" in line)
    returns_start = next(i for i, line in enumerate(lines) if "Returns" in line)
    del lines[note_start:returns_start]
    return "\n".join(lines)


def docstring(doc: str):
    def decorator(func: Callable):
        func.__doc__ = _modify_docstring(doc)
        return func

    return decorator


def init_logger() -> logging.Logger:
    logger = logging.getLogger("imlightgbm")
    logger.setLevel(logging.INFO)
    return logger


logger = init_logger()

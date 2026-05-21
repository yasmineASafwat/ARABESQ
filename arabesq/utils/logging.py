# arabesq/utils/logging.py
import logging
import os
from pathlib import Path
from typing import Optional, Union

def setup_logger(
    out_dir: Optional[Union[str, os.PathLike]] = None,
    name: str = "arabesq"
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fh = logging.FileHandler(out_dir / "run.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
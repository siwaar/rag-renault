"""Global Logger to use for all logs."""

from loguru import logger
from tqdm import tqdm

# Handle TQDM progress bar
logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)

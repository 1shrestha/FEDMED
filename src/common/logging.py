import logging
import sys


DEFAULT_LOG_LEVEL = "INFO"

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)s | "
    "%(name)s | "
    "%(message)s"
)


def setup_logging(level: str = DEFAULT_LOG_LEVEL) -> None:
    """
    Configure application-wide logging.

    This function should be called once when the FedMed application
    starts.
    """

    log_level = getattr(
        logging,
        level.upper(),
        logging.INFO,
    )

    logging.basicConfig(
        level=log_level,
        format=LOG_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Create or retrieve a named logger.

    Parameters
    ----------
    name:
        Name of the module requesting the logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    return logging.getLogger(name)
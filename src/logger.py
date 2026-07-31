"""
==========================================================
Employee Attrition Dashboard

logger.py

Author : Pramod Prakash Jadhav
==========================================================

Central logging configuration for the Employee Attrition
Dashboard project.
"""

import logging
from logging.handlers import RotatingFileHandler

from src.config import LOG_FILE, LOG_LEVEL


def get_logger(name: str = "EmployeeAttritionDashboard") -> logging.Logger:
    """
    Create and return a configured logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """

    logger = logging.getLogger(name)

    if logger.hasHandlers():
        return logger

    logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File Handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=2 * 1024 * 1024,   # 2 MB
        backupCount=5,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.propagate = False

    return logger


# ==========================================================
# Default Project Logger
# ==========================================================

logger = get_logger()

logger.info("Employee Attrition Dashboard Logger Initialized")

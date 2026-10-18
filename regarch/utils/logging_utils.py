"""Logging configuration for RegArch modules."""

import logging

import colorlog


def get_logger(level: int = logging.INFO, name: str = "regarch") -> logging.Logger:
    """Return an idempotently configured package logger.

    The function never clears handlers from the application's root logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    handler = next(
        (item for item in logger.handlers if item.get_name() == "regarch-console"),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        handler.set_name("regarch-console")
        handler.setFormatter(
            colorlog.ColoredFormatter(
                "%(log_color)s%(asctime)s %(levelname)s [%(process)d] %(message)s",
                log_colors={
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                },
            )
        )
        logger.addHandler(handler)
    handler.setLevel(level)
    return logger

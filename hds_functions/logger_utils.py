"""
Module name: logger_utils.py

A utility module for setting up and managing logging in a consistent, reusable way.

Functions:
- setup_logging: Configures a logger with console and file handlers.
- close_logging: Closes and removes all handlers from a logger.
- archive_logging: Archives the current log file by renaming it with a timestamp.
"""

import logging
import os
import shutil
from typing import Optional
from datetime import datetime
from .environment_utils import resolve_path


def setup_logging(
    log_dir: str,
    log_filename_base: str,
    logger_name: str,
    level: int = logging.INFO,
    formatter_str: str = '%(asctime)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """
    Sets up a logger with both file and console output.

    Args:
        log_dir (str): Path to the directory to store log files. Can be an absolute path or a relative path starting with './'.
        log_filename_base (str): Base name for the log file (without extension).
        logger_name (str): Identifier for the logger.
        level (int, optional): Logging level. Default is logging.INFO.
        formatter_str (str, optional): Format string for log output. Default is '%(asctime)s - %(levelname)s - %(message)s'.

    Returns:
        logging.Logger: Configured logger instance.

    Examples:
        ```python
        logger = setup_logging("./log_folder", "log_files", "my_logger")
        logger.info("App started.")
        logger.warning("Warning message.")
        ```
    """
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(level)
        logger.propagate = False  # Prevent log propagation to root logger

        resolved_log_dir = resolve_path(log_dir, repo=None)
        os.makedirs(resolved_log_dir, exist_ok=True)
        log_file_path = os.path.join(resolved_log_dir, f"{log_filename_base}.log")

        file_handler = logging.FileHandler(log_file_path, mode="a")
        file_handler.setLevel(level)
        file_handler.name = "file"

        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.name = "console"

        formatter = logging.Formatter(formatter_str)
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger


def close_logging(logger_name: str) -> None:
    """
    Closes and removes all handlers from a logger to free resources.

    Args:
        logger_name (str): Identifier for the logger to close.

    Returns:
        None

    Example:
        ```python
        close_logging("my_logger")
        ```
    """
    logger = logging.getLogger(logger_name)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def archive_logging(
    log_dir: str,
    log_filename_base: str,
    timestamp_format: str = "%Y%m%d_%H%M%S",
    separator: str = "_",
    verbose: bool = True
) -> Optional[str]:
    """
    Archives the current log file by copying it to a timestamped filename and deleting the original.

    Args:
        log_dir (str): Path to the directory to store log files. Can be absolute or relative.
        log_filename_base (str): Base name of the log file (without extension).
        timestamp_format (str, optional): Format used for the timestamp. Default is '%Y%m%d_%H%M%S'.
        separator (str, optional): Separator between the base name and timestamp. Default is '_'.
        verbose (bool, optional): If True, prints a message upon successful archiving. Default is True.

    Returns:
        Optional[str]: Path to the archived log file, or None if archiving fails.

    Raises:
        FileNotFoundError: If the original log file does not exist.
        RuntimeError: If archiving fails due to file system issues.

    Example:
        ```python
        archive_logging("./log_folder", "log_files", "%Y-%m-%d_%H-%M-%S")
        ```
    """
    resolved_log_dir = resolve_path(log_dir, repo=None)
    timestamp = datetime.now().strftime(timestamp_format)
    original_path = os.path.join(resolved_log_dir, f"{log_filename_base}.log")
    archived_path = os.path.join(resolved_log_dir, f"{log_filename_base}{separator}{timestamp}.log")

    if not os.path.exists(original_path):
        raise FileNotFoundError(f"Log file '{original_path}' does not exist.")

    try:
        shutil.copyfile(original_path, archived_path)
        os.remove(original_path)
        if verbose:
            print(f"Archived log: {archived_path}")
        return archived_path
    except (OSError, PermissionError) as e:
        raise RuntimeError(f"Failed to archive log: {e}")

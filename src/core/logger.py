"""Logging system for CompeteML with colored output"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import click


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output"""

    COLORS = {
        'DEBUG': 'cyan',
        'INFO': 'blue',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red'
    }

    def format(self, record):
        levelname = record.levelname
        message = super().format(record)

        # Add color to level name
        if levelname in self.COLORS:
            colored_level = click.style(levelname, fg=self.COLORS[levelname], bold=True)
            message = message.replace(levelname, colored_level)

        # Add success indicators
        if '✓' in message:
            message = message.replace('✓', click.style('✓', fg='green', bold=True))
        if '✗' in message:
            message = message.replace('✗', click.style('✗', fg='red', bold=True))

        return message


class CompeteMLLogger:
    """Centralized logging with colored console output"""

    def __init__(self, name: str = "CompeteML", log_dir: Optional[Path] = None, level: int = logging.INFO):
        self.name = name
        self.log_dir = log_dir or Path("outputs/logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers.clear()

        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_format = ColoredFormatter('%(levelname)s | %(message)s')
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)

        # File handler (no colors)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"competeml_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)

        self.log_file = log_file

    def debug(self, msg: str):
        self.logger.debug(msg)

    def info(self, msg: str):
        self.logger.info(msg)

    def success(self, msg: str):
        """Log success message"""
        self.logger.info(click.style(msg, fg='green'))

    def warning(self, msg: str):
        self.logger.warning(msg)

    def error(self, msg: str):
        self.logger.error(msg)

    def critical(self, msg: str):
        self.logger.critical(msg)

    def section(self, title: str):
        """Log a section header"""
        separator = "=" * 80
        header = f"\n{separator}\n{title.upper()}\n{separator}"
        self.logger.info(click.style(header, fg='cyan', bold=True))


def get_logger(name: str = "CompeteML") -> CompeteMLLogger:
    """Get or create a logger instance"""
    return CompeteMLLogger(name)

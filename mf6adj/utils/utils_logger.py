import logging
import os
import uuid


class _LoggerUtil:
    """Logger helper that configures console and optional file handlers."""

    def __init__(self, name, level, file_name=None):
        """Initialize a logger helper.

        Parameters
        ----------
        name : str
            Base logger name.
        level : str or int
            Logging level.
        file_name : str, optional
            Optional log file path.
        """
        instance_id = uuid.uuid4().hex[:8]
        name += f"-{instance_id}"

        self.name = name
        self.file_name = file_name

        self.logger = logging.getLogger(self.name)
        self._setup_logger(level)

        self.logger.info(f"Logger instance '{self.name}' created.")
        self.logger.info(f"Running from {os.getcwd()}")
        if file_name is not None:
            self.logger.info(f"Logger output written to '{file_name}'")

    def _get_logger_level(self, logging_level):
        """Normalize and apply the configured logging level.

        Parameters
        ----------
        logging_level : str or int
            Logging level to apply.
        """
        if isinstance(logging_level, str):
            if logging_level.upper() == "INFO":
                logging_level = logging.INFO
            elif logging_level.upper() == "DEBUG":
                logging_level = logging.DEBUG
            elif logging_level.upper() == "WARNING":
                logging_level = logging.WARNING
            elif logging_level.upper() == "ERROR":
                logging_level = logging.ERROR
            else:
                logging_level = logging.CRITICAL
        else:
            logging_level = max(logging_level, 0)

        self.level = logging_level
        self.logger.setLevel(self.level)

    def _setup_logger(self, level):
        """Configure logger handlers and formatters if needed.

        Parameters
        ----------
        level : str or int
            Logging level to apply when initializing handlers.
        """
        if not self.logger.handlers:
            self._get_logger_level(level)

            stream_handler = logging.StreamHandler()
            stream_formatter = logging.Formatter("%(asctime)s - %(message)s")
            stream_handler.setFormatter(stream_formatter)
            stream_handler.setLevel(logging.INFO)
            self.logger.addHandler(stream_handler)

            if self.file_name is not None:
                file_formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                file_handler = logging.FileHandler(self.file_name, mode="w")
                file_handler.setFormatter(file_formatter)
                file_handler.setLevel(self.level)
                self.logger.addHandler(file_handler)

    @property
    def isDebugLogger(self):
        """Return whether the logger is configured to emit debug messages.

        Returns
        -------
        bool
            ``True`` when the effective log level includes ``DEBUG``.
        """
        return self.logger.isEnabledFor(logging.DEBUG)

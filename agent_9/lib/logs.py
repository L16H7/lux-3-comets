import logging
import os


class FixedWidthFormatter(logging.Formatter):
    def format(self, record):
        record.levelname = f"{record.levelname:<8}"
        # Adjust filename to be 20 characters, padded with spaces if needed
        record.filename = f"{record.filename:<17}"  # Left-aligned, 20 chars wide
        # Adjust lineno to be 3 characters, padded with spaces in front if needed
        record.lineno = f"{record.lineno:>3}"  # Right-aligned, 3 chars wide
        return super().format(record)


def logging_init(team_id: int, step: int, logs_dir: str, is_logging: bool):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    if not is_logging:
        # Add a NullHandler to suppress logging output
        logging.basicConfig(level=logging.CRITICAL)
        logging.getLogger().addHandler(logging.NullHandler())
        return logging

    os.makedirs(logs_dir, exist_ok=True)
    fmt_string = '%(asctime)s.%(msecs)01d - %(levelname)s - %(filename)s L:%(lineno)s - %(message)s'
    date_string = '%M:%S'
    logging.basicConfig(
        filename=f'{logs_dir}/p{team_id}_s{step:03}.log',  # Name of the file to write logs to
        level=logging.DEBUG,
        format=fmt_string,
        datefmt=date_string
    )

    for handler in logging.root.handlers:
        handler.setFormatter(FixedWidthFormatter(fmt=fmt_string, datefmt=date_string))

    return logging

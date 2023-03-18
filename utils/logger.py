import loguru
import copy
import os
import datetime

def get_logger_and_log_path(log_root,
                            crt_date,
                            suffix,
                            use_date_prefix=False):
    """
    get logger and log path

    Args:
        log_root (str): root path of log
        crt_date (str): formated date name (Y-M-D)
        suffix (str): log save name
        use_date_prefix (bool): wheather to use date as prefix dir

    Returns:
        logger (loguru.logger): logger object
        log_path (str): current root log path (with suffix)
    """
    if use_date_prefix:
        log_path = os.path.join(log_root, crt_date, suffix)
    else:
        log_path = os.path.join(log_root, suffix)
    os.makedirs(log_path, exist_ok=True)

    logger_path = os.path.join(log_path, 'logfile.log')
    logger = loguru.logger
    fmt = '{time:YYYY-MM-DD at HH:mm:ss} | {message}'
    logger.add(logger_path, format=fmt)

    return logger, log_path

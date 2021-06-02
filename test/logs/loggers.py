from kowalsky.logs.loggers import LocalLogger
from kowalsky.logs.loggers import SqliteLogger


def test_local_logs():
    logger = LocalLogger(run_name='Bonjour')
    logger('Hello', 4)
    logger('Biba')
    logger.end_session()

def test_sqlite_logs():
    logger = SqliteLogger(run_name='a')
    logger('check1')
    logger('check2')
    logger('check3')
    logger.end_session()

if __name__ == '__main__':
    test_local_logs()

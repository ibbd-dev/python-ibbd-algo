"""
常用装饰器
"""
import signal
from functools import wraps


def Timeout(seconds, callback=None):
    """Add a timeout parameter to a function and return it.
    :param seconds: float: 超时时间
    :raises: HTTPException if time limit is reached
    """
    def decorate(function):
        def handler(signum, frame):
            if callback is None:
                raise Exception("Request timeout")
            else:
                callback()
        @wraps(function)
        def wrapper(*args, **kwargs):
            # 可以在函数调用的时候，指定超时参数
            old = signal.signal(signal.SIGALRM, handler)
            signal.setitimer(signal.ITIMER_REAL, seconds)
            # return function(*args, **kwargs)
            try:
                return function(*args, **kwargs)
            finally:
                # print('finally')
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old)
        return wrapper
    return decorate
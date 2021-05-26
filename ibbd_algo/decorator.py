"""
常用装饰器
"""
import signal
from functools import wraps


def Timeout(seconds, callback=None):
    """Add a timeout parameter to a function and return it.
    :param seconds: float: 超时时间
    :param callback: func|None: 回调函数，如果为None则会直接抛异常
    :raises: HTTPException if time limit is reached
    """
    def decorator(function):
        def handler(signum, frame):
            """超时处理函数"""
            if callback is None:
                raise Exception("Request timeout")
            else:
                # 超时回调函数
                callback()
        @wraps(function)
        def wrapper(*args, **kwargs):
            # SIGALRM: 时钟中断(闹钟)
            old = signal.signal(signal.SIGALRM, handler)
            # ITIMER_REAL: 实时递减间隔计时器，并在到期时发送 SIGALRM 。
            signal.setitimer(signal.ITIMER_REAL, seconds)
            try:
                return function(*args, **kwargs)
            finally:
                # 如果没有下面这两行，当function异常导致被捕获后，
                # 还可能会触发超时异常。
                # seconds=0: 意为清空计时器
                signal.setitimer(signal.ITIMER_REAL, 0)
                # 还原时钟中断处理
                signal.signal(signal.SIGALRM, old)
        return wrapper
    return decorator


if __name__ == "__main__":
    import time
    
    @Timeout(2)
    def test(i):
        time.sleep(i)
        return i
    try:
        v = test(2.5)
        print("Error")
    except Exception as e:
        print('ok: except timeout')
    test(1.5)
    def test(i):
        time.sleep(i)
        return i
    
    func = Timeout(2)(test)
    func(1.5)
    try:
        v = func(2.5)
        print("Error")
    except Exception as e:
        print('ok: except timeout')

    # 模拟函数异常
    def test(i):
        raise Exception('Raise in test')
        return i
    func = Timeout(1)(test)
    try:
        func(1.5)
    except Exception as e:
        print('try: ', e)
    time.sleep(2)
    print('end.')
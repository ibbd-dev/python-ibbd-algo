"""
常用装饰器
"""
import json
import signal
from functools import wraps
from .utils import md5


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


def CacheFunc(function):
    """对函数的返回结果进行缓存（使用redis）
    缓存时可以指定redis连接对象、key前缀、有效期等
    """

    @wraps(function)
    def wrapper(*args, _save_engine=None, _key_prefix='cf', _expire_second=30, **kwargs):
        """
        注意:
        1. 如果函数执行时间过长，如果使用redis连接可能会断开
        2. 注意kwargs中的参数不能和下面三个参数冲突
        3. 缓存保存引擎只需要支持两个接口：
           (1) get(key) -> value: 根据key获取获取的值
           (2) set(key, value, ex=seconds): 保存缓存并设置有效期
        :param _save_engine 缓存保存引擎，如redis.Redis()，如果该值为None则不进行缓存
        :param _key_prefix str 缓存key前缀
        :param _expire_second int 过期时间，单位：秒
        """
        if _save_engine is None:
            return function(*args, **kwargs)
        key = "%s:%s:%s" % (_key_prefix, function.__name__, md5(str(args) + str(kwargs)[8:24]))
        data = _save_engine.get(key)
        if data:
            return json.loads(data[1:])
        data = function(*args, **kwargs)
        _save_engine.set(key, "-"+json.dumps(data), ex=_expire_second)
        return data

    return wrapper



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

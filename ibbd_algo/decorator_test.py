# -*- coding: utf-8 -*-
#
# 单元测试文件: pytest decorator_test.py
# Author: caiyingyao
# Email: cyy0523xc@gmail.com
# Created Time: 2022-02-25
import time
import redis
from .decorator import CacheFunc

redis_pool = redis.ConnectionPool(
    host='192.168.1.242',
    decode_responses=True
)
redis_conn = redis.Redis(connection_pool=redis_pool)


def test_CacheFunc():
    """测试缓存"""
    @CacheFunc
    def func1(num):
        time.sleep(2)
        return num * 10

    @CacheFunc
    def func2(num=2):
        time.sleep(2)
        return {"num": num * 10}

    start = time.time()
    num = func1(2)
    assert time.time() - start > 1
    assert num == 20

    start = time.time()
    num = func1(2, redis_connect=redis_conn, key_prefix='test', expire_second=6)
    assert time.time() - start > 1
    assert num == 20

    # 这时已经有缓存了
    start = time.time()
    num = func1(2, redis_connect=redis_conn, key_prefix='test', expire_second=6)
    assert time.time() - start < 1
    assert num == 20

    start = time.time()
    num = func2(num=2, redis_connect=redis_conn, key_prefix='test', expire_second=60)
    assert time.time() - start > 1
    assert num['num'] == 20

    # 这时已经有缓存了
    start = time.time()
    num = func2(num=2, redis_connect=redis_conn, key_prefix='test', expire_second=6)
    assert time.time() - start < 1
    assert num['num'] == 20

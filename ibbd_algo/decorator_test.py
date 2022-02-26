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


def test_CacheFunc_int():
    """测试缓存"""
    @CacheFunc
    def func(num):
        time.sleep(2)
        return num * 10

    start = time.time()
    num = func(2)
    assert time.time() - start > 1
    assert num == 20

    start = time.time()
    num = func(2, _save_engine=redis_conn, _key_prefix='test1', _expire_second=6)
    assert time.time() - start > 1
    assert num == 20

    # 这时已经有缓存了
    start = time.time()
    num = func(2, _save_engine=redis_conn, _key_prefix='test1', _expire_second=6)
    assert time.time() - start < 1
    assert num == 20


def test_CacheFunc_dict():
    """测试缓存"""
    @CacheFunc
    def func(num=2):
        time.sleep(2)
        return {"num": num * 10}

    start = time.time()
    num = func(num=2, _save_engine=redis_conn, _key_prefix='test2', _expire_second=6)
    assert time.time() - start > 1
    assert num['num'] == 20

    # 这时已经有缓存了
    start = time.time()
    num = func(num=2, _save_engine=redis_conn, _key_prefix='test2', _expire_second=6)
    assert time.time() - start < 1
    assert num['num'] == 20


def test_CacheFunc_None():
    """测试缓存"""
    @CacheFunc
    def func():
        time.sleep(2)
        return None

    start = time.time()
    res = func(_save_engine=redis_conn, _key_prefix='test3', _expire_second=6)
    assert time.time() - start > 1
    assert res is None

    # 这时已经有缓存了
    start = time.time()
    res = func(_save_engine=redis_conn, _key_prefix='test3', _expire_second=6)
    assert time.time() - start < 1
    assert res is None

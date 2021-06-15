'''
通用的工具函数

Author: alex
Created Time: 2020年07月01日 星期三 09时42分12秒
'''
import numpy as np
from json import JSONEncoder
from concurrent import futures


def conc_map(func, ls_data, max_workers=None):
    """并发执行
    一个进程执行ls_data中的一个数据
    :param func function 需要执行的函数
    :param ls_data list 列表数据
    :param max_workers int|None 最大的worker数量，默认None则根据cpu的核数来定
    """
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(func, ls_data)
        return list(results)


def avg(arr):
    """普通的元组与列表等求均值
    如果是numpy则不需要使用该方法"""
    return sum(arr) / len(arr)


class NumpyJsonEncoder(JSONEncoder):
    """numpy数据序列化
    example:
        json.dump(np_data, write_file, cls=NumpyJsonEncoder)
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyJsonEncoder, self).default(obj)

'''
通用的工具函数

Author: alex
Created Time: 2020年07月01日 星期三 09时42分12秒
'''


def avg(arr):
    """普通的元组与列表等求均值
    如果是numpy则不需要使用该方法"""
    return sum(arr)/len(arr)

'''
字符串相关的通用函数

Author: alex
Created Time: 2020年07月03日 星期五 16时44分16秒
'''


def strQ2B(s):
    """字符串全角转半角"""
    return ''.join([Q2B(c) for c in s])


def Q2B(uchar):
    """单个字符 全角转半角"""
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0

    if inside_code < 0x0020 or inside_code > 0x7e:
        # 转完之后不是半角字符返回原来的字符
        return uchar
    return chr(inside_code)

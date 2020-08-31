# -*- coding: utf-8 -*-
#
# 安装程序
# Author: alex
# Created Time: 2018年04月02日 星期一 17时29分45秒
from distutils.core import setup


LONG_DESCRIPTION = """
IBBD常用算法集合
Algo
""".strip()

SHORT_DESCRIPTION = """常用算法集合""".strip()

DEPENDENCIES = [
    'numpy',
]

VERSION = '0.3.2'
URL = 'https://github.com/ibbd-dev/python-ibbd-algo'

setup(
    name='ibbd_algo',
    version=VERSION,
    description=SHORT_DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    url=URL,

    author='Alex Cai',
    author_email='cyy0523xc@gmail.com',
    license='Apache Software License',

    keywords='cluster optics dbscan 聚类',

    packages=['ibbd_algo'],
)

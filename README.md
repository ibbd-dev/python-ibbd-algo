# python-ibbd-algo

IBBD常用算法

## Install

```sh
# 安装依赖
pip3 install -r https://github.com/ibbd-dev/python-ibbd-algo/raw/master/requirements.txt

# 安装
pip3 install git+https://github.com/ibbd-dev/python-ibbd-algo.git
```

## 支持的算法列表

### Optics

scikit-learn中包含很多聚类的算法，但是在使用的过程中发现一个比较大的问题，如optics算法，不能自定义距离，只能造一个轮子。

```python
from ibbd_algo.optics import Optics

points = [
    np.array((1, 1)),
    np.array((1, 3)),
    np.array((2, 2)),
    np.array((4, 6)),
    np.array((5, 7)),
]

# 默认使用欧氏距离
optics = Optics(4, 2)
optics.fit(points)
clusters = optics.cluster(2)

for cluster in clusters:
    print(len(cluster))
    print(cluster)
```

可以使用自定义距离：

```python
def distance(point1, point2):
    data = [abs(a-b) for a, b in zip(point1, point2)]
    return sum(data)

optics = Optics(4, 2, distance=distance)
optics.fit(points)
```


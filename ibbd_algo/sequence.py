'''
序列相关算法

Author: alex
Created Time: 2020年06月03日 星期三 15时38分10秒
介绍文章：https://mp.weixin.qq.com/s?__biz=MzU3NDQ3MjI3Nw==&mid=2247484696&idx=1&sn=d8dc0d415a2f1eda30a324a6aefcf98b&chksm=fd30ac22ca472534e26ce7e4344db260c3a9c9ba7d9f6798581bdb3554aec094c4321a7b805d&token=521718035&lang=zh_CN#rd
'''
import numpy as np
import networkx as nx
from collections import Counter
from itertools import combinations
from fuzzywuzzy import fuzz
from diff_match_patch import diff_match_patch
from ibbd_algo.utils import conc_map

debug = False


def text_score(text1, text2, min_text_len=2):
    """计算两个文本的匹配得分
    :param min_text_len int 该参数可以用于减少偶然匹配的情况，也可以提升一些速度。但是该参数过大也会导致问题
    """
    if len(text1) < min_text_len and len(text2) < min_text_len:
        return 1.0      # TODO
    return fuzz.ratio(text1, text2) / 100


def text_score_dmp(text1, text2, min_text_len=2):
    """计算两个文本的匹配得分
    注意：这个算法会比text_score慢很多
    """
    if len(text1) < min_text_len and len(text2) < min_text_len:
        return 1.0      # TODO
    dmp = diff_match_patch()
    diffs = dmp.diff_main(text1, text2)
    dmp.diff_cleanupSemanticLossless(diffs)
    same_len = 0
    # 计算字节数是为了增加中文的权重
    for tag, text in diffs:
        if tag == 0:
            same_len += len(text.encode())

    # TODO 直接使用相同字符的数量作为得分应该是比较合理的
    return 2*same_len / (len(text1.encode())+len(text2.encode()))


def leave_repeat(arr):
    """留下重复的值"""
    arr = Counter(arr)
    return {key for key, val in arr.items() if val > 1}


def connected_components(edges):
    """找到连通的边"""
    G = nx.Graph()
    G.add_edges_from(edges)
    return nx.connected_components(G)


class Match:
    """序列匹配"""

    def __init__(self, seq1, seq2, score_func=text_score, max_workers=None, window=5):
        """两个序列的匹配
        :param seq1, seq2: list: 两个列表序列
        :param score_func: function(item1, item2): 得分函数，参数item1是seq1的元素，item2是seq2中的元素
        :param max_workers int|None: 并发的最大进程数量
        :param window int: 前后探索的窗口大小，窗口越大计算量越大。这个参数假设当前元素只和另一个序列的对应元素的前后若干个元素相关
        """
        # 计算得分
        len1, len2 = len(seq1), len(seq2)
        # 计算窗口的开始和结束位置
        start, end = -window, window
        if len2 >= len1:
            end += len2 - len1
        else:
            start += len2 - len1

        scores = np.zeros((len1, len2))
        for i, s1 in enumerate(seq1):
            # 一个元素通常只会和另一个序列中相邻的元素产生联系
            w_start, w_end = max(0, i+start), min(len2, i+end+1)
            scores[i][w_start:w_end] = conc_map(lambda j: score_func(s1, seq2[j]),
                                                range(w_start, w_end),
                                                max_workers=max_workers)

        self.scores = scores
        # print(scores[:3, :3])

    def match(self, min_score=0.2, sort_min_score=0.01, debug=False):
        """快速配对算法
        :return items: numpy.ndarray: [(idx1, idx2)]
        """
        scores = self.scores
        len1, len2 = scores.shape
        if len1 < 1 or len2 < 1:
            return np.array()
        if len1 == 1 and len2 == 1:
            return np.array([0, 0])
        is_T = False    # 是否转置
        if len1 > len2:
            is_T = True
            scores = scores.T

        # 每行最大值的索引
        max_index = np.argmax(scores, axis=1)
        eq_index = max_index == np.sort(max_index)
        len_index = len(max_index)
        if all(eq_index) and len_index == len(set(max_index)):
            # 顺序一致且每个值都不同
            data = [[i, v] for i, v in enumerate(max_index) if scores[i, v] >= sort_min_score]
            if is_T:
                data = [[v, i] for i, v in data]
            return np.array(data)

        # 顺序不一致或者存在相同的值
        if debug:
            print("begin...", flush=True)
        eq_index = np.zeros((len_index,), dtype=bool)
        min_arr, max_arr = self.get_min_max(max_index)
        for i in range(len_index):
            # 比前面的值都大，比后面的值都小
            if i > 0:
                if max_index[i] <= max_arr[i-1]:
                    eq_index[i] = True
                    continue
            if i < len_index-1:
                if max_index[i] >= min_arr[i+1]:
                    eq_index[i] = True

        if debug:
            print(max_index, flush=True)
            print(eq_index, flush=True)
        data = []
        is_true = False
        # self.min_score = min_score
        for (i, v), flag in zip(enumerate(max_index), eq_index):
            if not flag:
                is_true = False
                if scores[i, v] >= min_score:
                    data.append([i, v])
                if v == scores.shape[1]:     # 已经是最大的值
                    break
                continue
            if is_true:
                continue
            # 找到连续的True块
            is_true = True
            true_len = 0
            for j in range(i, len_index):
                if eq_index[j]:
                    true_len += 1
                else:
                    break

            # 在小区域[min_i:max_i, min_j:max_j]内进行匹配
            min_j = 0 if len(data) == 0 else data[-1][1]+1
            max_j = scores.shape[1] if i+true_len >= len_index else max_index[i+true_len]
            min_i, max_i = i, i+true_len
            self.scores = scores[min_i:max_i, min_j:max_j].copy()
            if debug:
                print("more match:", true_len, self.scores.shape, (min_i, max_i, min_j, max_j), flush=True)
                print(self.scores, flush=True)
            # tmp_data = self.more_match()
            tmp_data = self.match_old(min_score=min_score)
            if debug:
                print("tmp data: ", tmp_data)

            for tmp_i, tmp_v in tmp_data:
                data.append([tmp_i+min_i, tmp_v+min_j])

        if is_T:
            data = [[v, i] for i, v in data]
        return np.array(data)

    def get_min_max(self, array):
        """获取最小最大值列表"""
        n = array.shape[0]
        # 后面的最小值，前面的最大值
        min_arr, max_arr = np.zeros((n,), dtype=int), np.zeros((n,), dtype=int)
        min_arr[n-1], max_arr[0] = array[n-1], array[0]
        for i in range(1, n):
            max_arr[i] = array[i] if array[i] >= max_arr[i-1] else max_arr[i-1]
            j = n-i-1
            min_arr[j] = array[j] if array[j] <= min_arr[j+1] else min_arr[j+1]

        return min_arr, max_arr

    def match_old(self, min_score=None, force_comb=False, len_thr=8):
        """找到最优的匹配（旧版本）
        注意：匹配得到的顺序不能改变
        :param min_score: None|float: 允许匹配的最小得分，如果为None则不做判断
        :param force_comb: bool: 强制使用组合算法，注意如果队列元素比较多可能会很慢
        :param len_thr: int: 序列长度超过该值时，则不会使用组合算法，除非强制指定
        :return items: numpy.ndarray: [(idx1, idx2)]
        """
        self.min_score = min_score
        if all([self.scores.shape[0] < len_thr,
                self.scores.shape[1] < len_thr]) or force_comb:
            return self.less_match()

        # 当数据比较大时
        assert min_score is not None
        return self.more_match()

    def more_match(self):
        """当数据比较多时，使用该算法"""
        # 剪枝：其值却很小的边通常是没有意义的
        where_i, where_j = np.where(self.scores > self.min_score)
        len_j = len(where_j)
        if len_j == 0:
            return np.array([])
        if len_j == 1:
            return np.array([(where_i[0], where_j[0])])

        # 优化得分: 将位置影响整合到边的权重上
        for j, val_j in enumerate(where_j):
            # 正常来说，where_j是按顺序排序的
            # 如果前面有比当前值大，或者后面有比当前值小，这两种情况都是不常见的，可以减少其权重
            err_num = np.count_nonzero(where_j[:j] > val_j)
            err_num += np.count_nonzero(where_j[j:] < val_j)
            self.scores[where_i[j], val_j] *= (len_j-err_num)/(len_j)

        # 找到相连的边
        point_n = max(self.scores.shape)
        edges = [(i, j+point_n) for i, j in zip(where_i, where_j)]
        conn_nodes = connected_components(edges)
        data = []          # 返回值
        for nodes in conn_nodes:
            if len(nodes) == 2:
                # 只有一个关系
                a, b = min(nodes), max(nodes)
                data.append((a, b-point_n))
                continue
            data += self.parse_edges(edges, nodes, point_n)

        # 重排序
        data = sorted(data, key=lambda x: (x[0], x[1]))
        data = np.array(data)
        if data.shape[0] < 2:
            return data

        # 处理掉不符合顺序的关系
        # print(data[:, 0])
        # print(data[:, 1])
        del_ids = self.cal_del_ids(data[:, 1])    # 需要删除的
        if len(del_ids) > 0:
            all_ids = np.array([i for i in range(data.shape[0])
                                if i not in set(del_ids)], dtype=int)
            # TODO 直接删除可能未必是最好的方式
            data = data[all_ids]

        return data

    def cal_del_ids(self, data):
        """计算需要删除的id"""
        del_ids = []    # 需要删除的
        for j in range(len(data)-1):
            if data[j] < data[j+1]:
                continue     # 这是正常的
            # 保留j的损失
            loss_right = self.cal_loss_right(data[j+1:], data[j])
            # 保留j+1的损失
            loss_left = self.cal_loss_left(data[:j+1], data[j+1])
            # print('loss: ', loss_left, loss_right)
            if loss_right > loss_left:    # 右边损失比较大, 保留j+1
                del_ids += list(range(j-loss_left+1, j+1))
            else:      # 保留j
                del_ids += list(range(j+1, j+1+loss_right))

        # print('del: ', del_ids)
        # TODO 这里可能需要改进
        return del_ids

    def cal_loss_right(self, data, val):
        """计算右边损失"""
        loss = 0
        for i in data:
            if val > i:
                loss += 1
                continue
            break
        return loss

    def cal_loss_left(self, data, val):
        """计算左边损失"""
        loss = 0
        # print(data)
        for i in data[::-1]:
            if val < i:
                loss += 1
                continue
            break
        return loss

    def parse_edges(self, edges, nodes, j_diff):
        """处理特定顶点的边"""
        edges = [(i, j-j_diff) for i, j in edges if i in nodes]
        # print("==> ", edges)
        self.max_score = 0
        self.max_edges = []
        self.create_set(edges, [], 0, 0, set(), set())
        # print('max edges: ', self.max_edges)
        return self.max_edges

    def create_set(self, all_edges, edges, score, pos, set_i, set_j):
        """创建满足条件的集合"""
        for curr_pos in range(pos, len(all_edges)):
            i, j = all_edges[curr_pos]
            if len(set_i) > 0 and (i <= max(set_i) or j <= max(set_j)):
                # 不满足条件，直接进入下个元素
                continue

            t_edges = edges.copy()
            t_set_i, t_set_j = set_i.copy(), set_j.copy()
            t_edges.append((i, j))
            t_score = score + self.scores[(i, j)]
            t_set_i.add(i)
            t_set_j.add(j)
            self.create_set(all_edges, t_edges, t_score,
                            curr_pos+1, t_set_i, t_set_j)
            if t_score > self.max_score:
                # print('--> ', t_score, self.max_score)
                self.max_score = t_score
                self.max_edges = t_edges

    def less_match(self):
        """当数据比较小时，可以使用穷举匹配"""
        max_score = 0
        items = np.array([])
        max_num = min(self.scores.shape)
        for num in range(max_num, 0, -1):
            tmp_score, tmp_items = self.match_num(num)
            # print("less match: ", num, tmp_score, tmp_items)
            if tmp_score >= max_score and tmp_score > 0:
                # 两个空元素配对在一起，去掉之后得分不会改变，但是空元素不应该配对在一起
                items = tmp_items
                max_score = tmp_score
            # else:
                # 如果不能产生更好的值，则退出
                # break

        return items

    def fmt_items(self, items):
        """格式化items，并排序好，注意排序的时候应该保持顺序
        :return items list 如果没有对应的则对应值为-1
        """
        new_items = []
        if len(items) > 0:
            idx1, idx2 = set(items[:, 0]), set(items[:, 1])
            id_map = {id1: id2 for id1, id2 in items.tolist()}
        else:
            idx1, idx2 = set(), set()
            id_map = dict()

        for idx in range(self.scores.shape[0]):
            if idx not in idx1:
                new_items.append([idx, -1])
            else:
                new_items.append([idx, id_map[idx]])

        for idx in range(self.scores.shape[1]):
            if idx not in idx2:
                new_items = self.insert_item(new_items, [-1, idx])

        return new_items

    def insert_item(self, items, item):
        """插入一个item"""
        if item[1] == 0:
            items.insert(0, item)
            return items
        for idx, (_, val) in enumerate(items):
            if val == item[1]-1:
                items.insert(idx+1, item)
                break

        return items

    def match_num(self, num):
        """从两个序列中分别提取num个元素进行匹配"""
        # print('match num: ', num)
        max_score = 0
        comb_match = None
        # 提取两个序列的下标子集合
        comb1 = combinations(range(self.scores.shape[0]), num)
        comb2 = list(combinations(range(self.scores.shape[1]), num))
        for comb_i in comb1:
            for comb_j in comb2:
                # 计算这两个集合的得分
                tmp_score = self.cal_comb_score(comb_i, comb_j)
                if tmp_score is None:
                    continue
                if tmp_score >= max_score:
                    max_score = tmp_score
                    comb_match = (comb_i, comb_j)

        # 生成配对items
        if comb_match is None:
            return 0, np.array([])
        # print(comb_match)
        items = np.array((comb_match[0], comb_match[1])).T
        return max_score, items

    def cal_comb_score(self, comb_i, comb_j):
        """计算集合得分"""
        where = (np.array(comb_i, dtype=int), np.array(comb_j, dtype=int))
        scores = self.scores[where]
        if self.min_score is not None and np.min(scores) < self.min_score:
            return None
        return np.sum(scores)


if __name__ == '__main__':
    seq1 = ['中国人民']
    seq2 = ['中国人民呀', '人民共和国']
    match = Match(seq1, seq2)
    res = match.match()
    assert res.tolist() == [[0, 0]]
    res = match.fmt_items(res)
    assert res == [[0, 0], [-1, 1]]

    seq1 = ['中国人民']
    seq2 = ['人民共和国', '中国人民呀']
    res = Match(seq1, seq2).match()
    assert res.tolist() == [[0, 1]]

    seq2 = ['中国人民']
    seq1 = ['人民共和国', '中国人民呀']
    res = Match(seq1, seq2).match()
    assert res.tolist() == [[1, 0]]

    seq1 = ['中国人民', '广东省广州市']
    seq2 = ['人民共和国', '中国人民呀', '广东广州天河']
    res = Match(seq1, seq2).match()
    match = Match(seq1, seq2)
    res = match.match()
    assert res.tolist() == [[0, 1], [1, 2]]
    res = match.fmt_items(res)
    assert res == [[-1, 0], [0, 1], [1, 2]]

    seq1 = ['中国人民', '广东省广州市']
    seq2 = ['中国人民呀', '人民共和国', '广东广州天河']
    match = Match(seq1, seq2)
    res = match.match()
    assert res.tolist() == [[0, 0], [1, 2]]
    res = match.fmt_items(res)
    assert res == [[0, 0], [-1, 1], [1, 2]]

    seq1 = ['中国人民', '广东省广州市']
    seq2 = []
    match = Match(seq1, seq2)
    res = match.match()
    assert res.tolist() == []
    res = match.fmt_items(res)
    assert res == [[0, -1], [1, -1]]

    seq1 = []
    seq2 = ['中国人民', '广东省广州市']
    match = Match(seq1, seq2)
    res = match.match()
    assert res.tolist() == []
    res = match.fmt_items(res)
    assert res == [[-1, 0], [-1, 1]]

    seq1 = ['中国人民', 'abcd', '广东广州市']
    seq2 = ['中国人', 'defijk', '广东省广州市']
    match = Match(seq1, seq2)
    res = match.match()
    assert res.tolist() == [[0, 0], [1, 1], [2, 2]]
    res = match.match(min_score=0.5)
    print(res.tolist())
    assert res.tolist() == [[0, 0], [1, 1], [2, 2]]
    res = match.fmt_items(res)
    assert res == [[0, 0], [1, 1], [2, 2]]

    seq1 = ['迪奥科技有限公司', '迪奥科技', '电子档案质检系统五期',
            '(暨合同内容比对)', '解决方案']
    seq2 = ['迪奥科技有限公司', '迪奥科技电子档案质检系统', '暨合同内容比对)',
            '解决方案', '1.引言']
    match = Match(seq1, seq2)
    res = match.match(min_score=0.55)
    assert res.shape == (4, 2)

    seq1 = ['2', '迪奥科技有限公司', '1.引言', '1.1.编写目的', '明确迪奥科技电子档案质检系统五期(合同比对)的功能范围、功能模块细', '节、系统处理流程,为项目设计人员提供进一步设计依据、为项目开发人员提供', '开发依据、为测试人员提供测试依据、为客户的项目验收提供验收依据', '1.2.目标与背景', '1.2.1.项目背景', '1.合同是防范法律风险的必要程序,尤其在金融行业,合同在风险防控、合', '规管理、客户权益等方面尤为重要。业务合同通常条款详细,且多为制式合同',
            '为提高合同签署效率,防止合同被另一方恶意修改,或者合同被伪造等,需要对', '合同的全部文字条款做内容确认,合同文本审核的工作量非常大。', '2-1文本识别比对需求', '2.传统的人工审核方式不仅效率低下,且容易受审核人员业务素养、体力', '2', '精神状态等因素的影响,一旦审核出现疏漏差错,损失将是巨大的', '目前迪奥科技虽然采用了合同添加水印和二维码等方式进行版本控制,但仍存在', '2']
    seq2 = ['迪奥科技有限公司', '1.1.编写目的', '明确迪奥科技电子档案质检系统六期的功能范围、功能模块细节、系统处理', '流程,为项目设计人员提供进一步设计依据、为项目开发人员提供开发依据、为', '测试人员提供测试依据、为客户的项目验收提供验收依据.', '1.2.目标与背景', '1.2.1.项目背景', '1.合同是防范法律风险的必要程序,尤其在金融行业,合同在风险防控、合',
            '规管理、客户权益等方面尤为重要。业务合同通常条款详细,包括文字,表格,', '盖章等的识别和比对,且多为制式合同,为提高合同签署效率,防止合同被另一', '方恶意修改,或者合同被伪造等,需要对合同的全部文字条款做内容确认,合同', '文本审核的工作量非常大', '图2-1文本识别比对需求', '2.传统的人工审核方式不仅效率低下,且容易受审核人员业务素养、体力、精', '神状态等因素的影响,一旦审核出现疏漏差错,损失将是巨大的.', '这里是新增的一行', '2']
    match = Match(seq1, seq2)
    res = match.match(min_score=0.65)
    for pos in range(len(res)):
        i, j = res[pos]
        print('-> ', i, j)
        print('  ', seq1[i])
        print('  ', seq2[j])

    seq1 = ["aa", "bb", "cc"]
    seq2 = ["aa", "cc", "bc", "dd"]
    match = Match(seq1, seq2)
    res = match.match(min_score=0.1, debug=True)
    print(res)
    assert res.shape == (2, 2)
    assert res[0].tolist() == [0, 0]
    assert res[1].tolist() == [2, 1]

    seq1 = ["aa", "cc", "bc", "dd"]
    seq2 = ["aa", "bb", "cc"]
    match = Match(seq1, seq2)
    res = match.match(min_score=0.1)
    print(res)
    assert res.shape == (2, 2)
    assert res[0].tolist() == [0, 0]
    assert res[1].tolist() == [1, 2]

    debug = True
    seq1 = ["bb", "cc"]
    seq2 = ["cc", "cc", "bd"]
    match = Match(seq1, seq2)
    match.min_score = 0.1
    res = match.more_match()
    print("res:", res)
    res = match.match_old(min_score=0.1)
    print("res:", res)
    assert res.shape == (1, 2)
    assert res[0].tolist() == [1, 1]

    seq1 = ["aa", "cc", "cc", "bd"]
    seq2 = ["aa", "bb", "cc"]
    match = Match(seq1, seq2)
    res = match.match(min_score=0.1, debug=True)
    print(res)
    assert res.shape == (2, 2)
    assert res[0].tolist() == [0, 0]
    assert res[1].tolist() in ([1, 2], [2, 2])
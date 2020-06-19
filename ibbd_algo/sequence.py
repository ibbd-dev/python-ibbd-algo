'''
序列相关算法

Author: alex
Created Time: 2020年06月03日 星期三 15时38分10秒
'''
import numpy as np
from itertools import combinations
from diff_match_patch import diff_match_patch


def text_score(text1, text2):
    """计算两个文本的匹配得分"""
    dmp = diff_match_patch()
    diffs = dmp.diff_main(text1, text2)
    dmp.diff_cleanupSemanticLossless(diffs)
    same_len = 0
    # 计算字节数是为了增加中文的权重
    for tag, text in diffs:
        if tag == 0:
            same_len += len(text.encode())

    return 2*same_len / (len(text1.encode())+len(text2.encode()))


class Match:
    def __init__(self, seq1, seq2, score_func=text_score):
        """两个序列的匹配
        :param seq1, seq2: list: 两个列表序列
        :param score_func: function(item1, item2): 得分函数，参数item1是seq1的元素，item2是seq2中的元素
        """
        # 计算得分
        scores = dict()
        for i, s1 in enumerate(seq1):
            for j, s2 in enumerate(seq2):
                scores[(i, j)] = score_func(s1, s2)

        self.scores = scores
        self.seq1 = seq1
        self.seq2 = seq2

    def match(self, min_score=None):
        """找到最优的匹配
        注意：匹配得到的顺序不能改变
        :param min_score 允许匹配的最小得分，如果为None则不做判断
        :return items: list: [(idx1, idx2)]
        """
        self.min_score = min_score
        max_score = 0
        items = np.array([])
        max_num = min(len(self.seq1), len(self.seq2))
        for num in range(1, max_num+1):
            tmp_score, tmp_items = self.match_num(num)
            if tmp_score > max_score:
                items = tmp_items
                max_score = tmp_score

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

        for idx in range(len(self.seq1)):
            if idx not in idx1:
                new_items.append([idx, -1])
            else:
                new_items.append([idx, id_map[idx]])

        for idx in range(len(self.seq2)):
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
        """从seq1中提取num个元素进行匹配"""
        max_score = 0
        comb_match = None
        comb1 = list(combinations(range(len(self.seq1)), num))
        comb2 = list(combinations(range(len(self.seq2)), num))
        for comb_i in comb1:
            for comb_j in comb2:
                tmp_score = self.cal_comb_score(comb_i, comb_j)
                if tmp_score is None:
                    continue
                if tmp_score >= max_score:
                    max_score = tmp_score
                    comb_match = (comb_i, comb_j)

        # 生成配对items
        if comb_match is None:
            return 0, []
        # print(comb_match)
        items = np.array((comb_match[0], comb_match[1])).T
        return max_score, items

    def cal_comb_score(self, comb_i, comb_j):
        """计算集合得分"""
        scores = [self.scores[(i, j)] for i, j in zip(comb_i, comb_j)]
        if self.min_score is not None and min(scores) < self.min_score:
            return None
        return sum(scores)


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
    assert res.tolist() == [[0, 0], [2, 2]]
    res = match.fmt_items(res)
    assert res == [[0, 0], [-1, 1], [1, -1], [2, 2]]

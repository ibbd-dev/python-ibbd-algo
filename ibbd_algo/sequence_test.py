'''
sequence测试

Author: alex
Created Time: 2021年04月30日 星期五 10时01分47秒
'''
import sys
import time
import json
from sequence import Match, text_score

# python .\sequence_test.py D:\git\src\git.ibbd.net\gf\doc-compare\tests\debug_match_pages_data.json
with open(sys.argv[1], encoding='utf8') as f:
    s_lines, d_lines = json.load(f)

window = 1
start = time.time()
match = Match(s_lines, d_lines, score_func=lambda l1, l2: text_score(l1, l2, min_text_len=5), window=window)
print('===> Match init: ', time.time()-start, flush=True)
matches = match.match(min_score=0.1, debug=True)
print('===> Match Page: %d' % len(matches), time.time()-start, flush=True)

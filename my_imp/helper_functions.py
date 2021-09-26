import math

import numpy as np
import mmap
import os
from tqdm import tqdm
import numba as nb
from numba import cuda, float32
from bisect import bisect_left


@nb.njit(fastmath=True, parallel=True)
def inverse(A_inv, B):
    '''
    reference: https://math.stackexchange.com/questions/17776/inverse-of-the-sum-of-matrices
    :param A_inv: inverse of A
    :param B: dot product of context vector
    :return: updated A_inv
    '''
    temp = np.dot(B, A_inv)
    g = np.trace(temp)
    inverse = A_inv - (np.dot(A_inv, temp)) * (1 / (1 + g))
    return inverse


def get_num_lines(file_path):
    fp = open(file_path, 'r+')
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines


def parseLine(line):
    line = line.split("|")

    tim, articleID, click = line[0].strip().split(" ")
    tim, articleID, click = int(tim), int(articleID), int(click)
    user_features = np.array([float(x.strip().split(':')[1])
                              for x in line[1].strip().split(' ')[1:]])

    pool_articles = [l.strip().split(" ") for l in line[2:]]
    pool_articles = np.array(
        [[int(l[0])] + [float(x.split(':')[1]) for x in l[1:]] for l in pool_articles])
    return tim, articleID, click, user_features, pool_articles


def num_articles(folder):
    articles = []
    i = 0
    for root, dirs, files in os.walk(folder):
        for filename in files:
            i += 1
            f = open(os.path.join(root, filename), "r")
            print(f"File number {i}")
            max_ = get_num_lines(os.path.join(root, filename))
            for line_data in tqdm(f, total=max_):
                tim, articleID, click, user_features, pool_articles = parseLine(
                    line_data)
                for article in pool_articles:
                    if int(article[0]) not in articles:
                        articles.append(int(article[0]))
    return articles


def get_all_articles():
    # All days
    articles = [109498, 109509, 109508, 109473, 109503, 109502, 109501, 109492, 109495, 109494, 109484, 109506,
                109510, 109514, 109505, 109515, 109512, 109513, 109511, 109453, 109519, 109520, 109521, 109522,
                109523, 109524, 109525, 109526, 109527, 109528, 109529, 109530, 109534, 109532, 109533, 109531,
                109535, 109536, 109417, 109542, 109538, 109543, 109540, 109544, 109545, 109546, 109547, 109548,
                109550, 109552, 109553, 109551, 109554, 109555, 109518, 109556, 109476, 109557,
                109558, 109541, 109559, 109560, 109561, 109562, 109563, 109564, 109565, 109566, 109567, 109568,
                109569, 109570, 109571, 109572, 109517, 109573, 109539, 109574, 109575, 109576, 109577, 109578,
                109579, 109580, 109581, 109582, 109583, 109585, 109586, 109584, 109587, 109589, 109588, 109591,
                109592, 109593, 109594
        , 109595, 109596, 109597, 109598, 109600, 109599, 109601, 109606, 109605, 109607, 109608, 109603,
                109602, 109609, 109610, 109613, 109615, 109617, 109614, 109611, 109612, 109618, 109619, 109620,
                109621, 109622, 109623, 109624, 109625, 109626, 109627, 109628, 109629, 109630, 109631, 109616,
                109632, 109633, 109634, 109638, 109635, 109637, 109640, 109641, 109636, 109650, 109646, 109644,
                109652, 109653, 109647, 109654, 109655, 109656, 109657, 109658, 109659, 109660,
                109662, 109661, 109663, 109664, 109667, 109668, 109666, 109665, 109669, 109670, 109671, 109648,
                109674, 109676, 109677, 109678, 109679, 109680, 109681, 109682, 109683, 109687, 109688, 109689,
                109673, 109690, 109691, 109692, 109693, 109694, 109695, 109696, 109697, 109651, 109698, 109699,
                109700, 109701, 109702, 109703, 109704, 109705, 109706, 109675, 109707, 109708, 109709, 109710,
                109716, 109717, 109718, 109719, 109720, 109721, 109722, 109723, 109724, 109728,
                109725, 109726, 109727, 109729, 109732, 109711, 109735, 109736, 109737, 109742, 109743, 109734,
                109744, 109745, 109746, 109747, 109748, 109749, 109714, 109686, 109752, 109753, 109754, 109684,
                109755, 109756, 109757, 109758, 109759, 109760, 109741, 109761, 109762, 109763, 109766, 109764,
                109740, 109767, 109731
        , 109769, 109770, 109771, 109772, 109765, 109773, 109774, 109775, 109776, 109777, 109778, 109779,
                109780, 109781, 109782, 109783, 109730, 109784, 109785]

    # First two days
    articles = [109498, 109509, 109508, 109473, 109503, 109502, 109501, 109492, 109495, 109494, 109484, 109506, 109510,
                109514, 109505, 109515, 109512, 109513, 109511, 109453, 109519, 109520, 109521, 109522, 109523, 109524,
                109525, 109526, 109527, 109528, 109529, 109530, 109534, 109532, 109533, 109531, 109535, 109536, 109417,
                109542, 109538, 109543, 109540, 109544, 109545, 109546, 109547, 109548, 109550, 109552, 109553, 109551,
                109554, 109555, 109518, 109556, 109476, 109557, 109558, 109541, 109559, 109560, 109561, 109562, 109563,
                109564, 109565, 109566, 109567, 109568, 109569, 109570, 109571]
    articles = sorted(articles)
    return articles


def get_near_psd(A):
    C = (A + A.T) / 2
    eigval, eigvec = np.linalg.eig(C)
    eigval[eigval < 0] = 0

    return eigvec.dot(np.diag(eigval)).dot(eigvec.T)


def is_positive_definate(A):
    '''
    if the matrix has a choleskey decomposition solution then it will be positive definite.
    '''
    if np.array_equal(A, A.T):  # checking if the matrix is symmetric
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def check_readline(file):
    import math
    f = open(file,"r")
    max_ = get_num_lines(file)
    for i in tqdm(range(math.ceil(max_/5))):
        lines = []
        for j in range(5):
                lines.append(f.readline())
        for line in lines:
            if len(line)!=0:
                print(line)

def makeContext(pool_articles, user_features, articles):
    context = {}
    for article in pool_articles:
        if len(article) == 7 and len(user_features) == 6:
            all_zeros = np.zeros(len(articles) * 6 + 6)
            for i in range(len(articles)):
                if articles[i] == int(article[0]):
                    all_zeros[i * 6:i * 6 + 6] = user_features
            all_zeros[len(articles) * 6:] = article[1:]
            context[int(article[0])] = all_zeros
    return context


@nb.njit(fastmath=True, parallel=True)
def random_sampling(mean, cov, d, n,seed):
    np.random.seed(seed)
    L = np.linalg.cholesky(cov)
    u = np.random.normal(loc=0, scale=1, size=d * n)
    new_mean = mean + np.dot(u, L).flatten()
    return new_mean


def bin_search(a, x):
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    else:
        return -1


if __name__ == "__main__":
    # print(num_articles("data/R6A_spec"))
    check_readline("data/R6A.README")

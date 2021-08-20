import numpy as np
import mmap
import os
from tqdm import tqdm
def inverse(A_inv, B):
    '''
    reference: https://math.stackexchange.com/questions/17776/inverse-of-the-sum-of-matrices
    :param A_inv: inverse of A
    :param B: dot product of context vector
    :return: updated A_inv
    '''
    temp = np.matmul(B,A_inv)
    g = np.trace(temp)
    inverse =  A_inv - (np.matmul(A_inv,temp)) * (1 / (1 + g))
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
            i+=1
            f = open(os.path.join(root,filename), "r")
            print(f"File number {i}")
            max_ = get_num_lines(os.path.join(root,filename))
            for line_data in tqdm(f,total=max_):
                tim, articleID, click, user_features, pool_articles = parseLine(
                    line_data)
                for article in pool_articles:
                    if int(article[0]) not in articles:
                        articles.append(int(article[0]))
    return articles
if __name__ == "__main__":
    A = np.random.rand(3, 3)
    A_inv = np.linalg.inv(A)
    B = np.random.rand(3, 3)
    print(A)
    print(B)
    print(np.linalg.inv(A + B))
    print(inverse(A_inv, B))
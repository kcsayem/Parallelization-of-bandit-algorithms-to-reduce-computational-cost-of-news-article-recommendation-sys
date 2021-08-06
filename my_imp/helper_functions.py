import numpy as np

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

if __name__ == "__main__":
    A = np.random.rand(3, 3)
    A_inv = np.linalg.inv(A)
    B = np.random.rand(3, 3)
    print(A)
    print(B)
    print(np.linalg.inv(A + B))
    print(inverse(A_inv, B))
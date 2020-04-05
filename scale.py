from scipy.sparse import csr_matrix


def scale(s, r, maxS, minS):
    if isinstance(s, csr_matrix):
        s = s.toarray()

    if maxS == minS and maxS == 1:
        s = 2 * r * s - r
    else:
        s = (s - minS) / (maxS - minS)
        s = 2 * r * s - r
    return s

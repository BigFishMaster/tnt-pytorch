import sklearn.cluster.DBSCAN
b = globals()
b = [(k, v) for k, v in b.items() if "__" not in k and callable(v)]

def test():
    pass

def test1():
    pass


class test2:
    pass


def test3():
    a = globals()
    a = [(k, v) for k, v in a.items() if "__" not in k and callable(v)]
    print("globals:", a)

test3()
print(b)

from functools import partial
def test(a, b, c):
    h = locals()
    print(h)


def test1(output, target, loss):
    h = locals()
    print(h)


class Stats():
    def __init__(self, a, b, c=0):
        print(a, b, c)

def test2():
    fn = partial(Stats, c=-11)
    fn(1,2)

if __name__ == "__main__":
    #test1(2, 3, 5)
    test2()

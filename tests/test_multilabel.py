import random


def test():
    num = 1000
    class_num = 100
    labels = list(range(class_num))
    lens = list(range(1, 10))
    image_name = "data/sample.jpg"
    fout = open("data/multilabel_data/valid.txt", "w", encoding="utf8")
    for i in range(num):
        random.shuffle(labels)
        random.shuffle(lens)
        ll = lens[0]
        labs = labels[:ll]
        str_lab = " ".join([str(l) for l in labs])
        fout.write(image_name + " " + str_lab + "\n")
    fout.close()


if __name__ == "__main__":
    test()

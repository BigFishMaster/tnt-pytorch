class Landmark:

    def __init__(self):
        super(Landmark, self).__init__()
        self.train_file = "./data/train_data.txt"
        self.query_file = "./data/query_data.txt"
        self.gallery_file = "./data/gallery_data.txt"

        self.train = self._load_txt(self.train_file)
        self.query = self._load_txt(self.query_file)
        self.gallery = self._load_txt(self.gallery_file)
        print("data info: train={}, query={}, gallery={}".format(
            len(self.train), len(self.query), len(self.gallery)))

    def _load_txt(self, path):
        input = open(path, "r").readlines()
        dataset = []
        for i, item in enumerate(input):
            tts = item.strip().split()
            path, label, camera = tts
            label = int(label)
            camera = int(camera)
            dataset.append((path, label, camera))
        return dataset


class LandmarkInference:

    def __init__(self, test_file):
        super(LandmarkInference, self).__init__()
        self.infer = self._load_txt(test_file)

    def _load_txt(self, path):
        input = open(path, "r").readlines()
        dataset = []
        for i, item in enumerate(input):
            path = item.strip()
            dataset.append(path)
        return dataset

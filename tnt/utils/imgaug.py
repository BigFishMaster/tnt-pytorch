import os
import Augmentor
from Augmentor.ImageUtilities import AugmentorImage


class Augment:
    def __init__(self, path_to_file, save_folder, repeat_target=50):
        self.p = Augmentor.Pipeline()
        input = open(path_to_file, "r").readlines()
        augmentor_images = []
        dic1 = {}
        for i, item in enumerate(input):
            image_path, label = item.strip().split()
            if label not in dic1:
                dic1[label] = []
            dic1[label].append(image_path)
        for i, key in enumerate(dic1):
            values = dic1[key]
            num = repeat_target // len(values) + 1
            values = values * num
            for v in values:
                a = AugmentorImage(image_path=v, output_directory=save_folder)
                a.class_label = "none"
                a.file_format = "jpg"
                augmentor_images.append(a)

        print("Image list length:", len(augmentor_images))
        self.p.augmentor_images = augmentor_images

        self.p.rotate(probability=0.5, max_left_rotation=15, max_right_rotation=15)
        self.p.zoom(probability=0.5, min_factor=1.1, max_factor=1.2)
        self.p.shear(probability=0.5, max_shear_left=10, max_shear_right=10)
        self.p.crop_centre(probability=0.5, percentage_area=0.9)
        self.p.skew_tilt(probability=0.5, magnitude=0.2)
        self.p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)

    def sample(self):
        self.p.sample(n=0)


def test():
    a = Augment("data.txt", "./output/", repeat_target=10)
    a.sample()


if __name__ == "__main__":
    test()
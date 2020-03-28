from PIL import Image

# RGB, (W, H, C)
im = Image.open("data/sample.jpg")
print(im)
print(im.size, im.format, im.mode)
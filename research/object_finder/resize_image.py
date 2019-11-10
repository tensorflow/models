from PIL import Image
import os

def resize_image(directory, size):
    for image in os.listdir(directory):
        img = Image.open(directory+image)
        resized_img = img.resize(size, Image.LANCZOS)
        resized_img.save(directory+image)

if __name__ == "__main__":
    resize_image("dataset/",(800,600))
    print("Executed")
from sys import argv
import torch
import torch.nn.parallel
import models.fashion as models
import transforms
import numpy as np
import os
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

type_list = [
    "T-shirt/top", 
    "Trouser", 
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

def imageprepare(argv):
    """
    This function returns the pixel values.
    The imput is a png file location.
    """
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels

    if width > height:  # check which dimension is bigger
        # Width is bigger. Width becomes 20 pixels.
        nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
        if (nheight == 0):  # rare case but minimum is 1 pixel
            nheight = 1
            # resize and sharpen
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
        newImage.paste(img, (4, wtop))  # paste resized image on white canvas
    else:
        # Height is bigger. Heigth becomes 20 pixels.
        nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
        if (nwidth == 0):  # rare case but minimum is 1 pixel
            nwidth = 1
            # resize and sharpen
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
        newImage.paste(img, (wleft, 4))  # paste resized image on white canvas

    # newImage.save("sample.png

    tv = list(newImage.getdata())  # get pixel values

    # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    return tva

def plot(images):
    plt.figure(figsize=(10,10))
    plt.imshow(images[0].permute(1, 2, 0))
    plt.show()

def main(argv):
    path_name = argv[1]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "./checkpoint/model_best.pth.tar"

    preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    model = models.__dict__["resnet"](
                        num_classes=10,
                        depth=20,
                    )
    model = torch.nn.DataParallel(model).to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path)["state_dict"], strict=False)
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))["state_dict"], strict=False)
    model.eval()

    img = np.asarray(imageprepare(os.path.join(path_name))).reshape(28, 28, 1)
    img_tensor = preprocess(img).unsqueeze(0).float()
    with torch.no_grad():
        result = model(img_tensor.to(device)).argmax(dim=-1).cpu().numpy()
    print(type_list[result[0]])

if __name__ == '__main__':
    main(argv)

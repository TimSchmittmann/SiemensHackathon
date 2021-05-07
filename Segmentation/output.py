import torch
import network
import utils
from PIL import Image
from torchvision import transforms
from PIL import ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: %s" % device)

#load model
model_map = {
    'deeplabv3_resnet50': network.deeplabv3_resnet50,
    'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
    'deeplabv3_resnet101': network.deeplabv3_resnet101,
    'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
    'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
    'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
}
model = model_map['deeplabv3plus_resnet101'](num_classes=19, output_stride=16)
model.load_state_dict(torch.load("best_deeplabv3plus_resnet101_cityscapes_os16.pth.tar",map_location=torch.device('cpu'))["model_state"])
model.eval()
model.to(device)

#utils

denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

#prepare data
#for file in glob.glob(os.path.join("planar2/","*.jpg")):


filedir = sorted(os.listdir('planar2/'))
filedir = ['planar2/'+x for x in filedir]
print(filedir)

for file in filedir:
    title = file.title().replace(".Jpg","").replace("Planar2/","")
    if os.path.isfile("{}.png".format(title)) == True:
        continue

    images = Image.open(file)
    #title = file.title().replace(".Jpg","").replace("Planar2/","")
    images = images.convert("RGB")
    print(title)

    #sharpen
    #images = images.filter(ImageFilter.SHARPEN)
    #images = images.filter(ImageFilter.SHARPEN)
    #enhancer = ImageEnhance.Contrast(images)
    #factor = 1.5 #increase contrast
    #images = enhancer.enhance(factor)
    #plt.imshow(images)

    #images = images.resize((1232,1028))
    images = images.resize((1540,1285))
    #images = images.resize((200,200))

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    images = preprocess(images)

    #data in model
    images = images.to(device, dtype=torch.float32)
    images = images.unsqueeze_(0)
    images.shape

    #inference
    output = model(images)
    output = output[0]
    output = output.argmax(0)

    # plot the semantic segmentation predictions of 19 classes in each color
    inf = output.byte().cpu().numpy()

    #put inf in list with class
    lengthx,lengthy = inf.shape
    length = lengthx*lengthy
    list = []
    #0,0 is top, left
    x = 0
    y = 0
    tuple = (0,x,y)

    for a in inf:
        for b in a:
            if b == 1:
                b = 0
            tuple = (b,x,y)
            list.append(tuple)
            x += 1
        y+=1
        x=0
    with open("info{].txt".format(title)) as f:
        f.write(tuple)
        f.close()

    """
        if x<lengthx:
            tuple = (a,x,y)
            list.append(tuple)
            x += 1
        else:
            x=0
            y+=1
            tuple = (a,x,y)
            list.append(tuple)
    """
    #print(list)

    r = Image.fromarray(inf)
    r = r.resize((2464,2056))
    #show stuff
    #plt.imshow(r)
    #plt.show()
    plt.imsave("{}.png".format(title), r)

    #r.save("{}.jpg".format(title))


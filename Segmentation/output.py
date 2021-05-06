import torch
import network
import utils
from PIL import Image
from torchvision import transforms
from PIL import ImageFilter, ImageEnhance
import matplotlib.pyplot as plt

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
images = Image.open("sideview_000000_001594.jpg")
images = images.convert("RGB")
#sharpen

images = images.filter(ImageFilter.SHARPEN)
images = images.filter(ImageFilter.SHARPEN)
#enhancer = ImageEnhance.Contrast(images)
#factor = 1.5 #increase contrast
#images = enhancer.enhance(factor)
plt.imshow(images)

#images = images.resize((1232,1028))
images = images.resize((1540,1285))

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
images = preprocess(images)

#data in model
images = images.to(device, dtype=torch.float32)
images = images.unsqueeze_(0)

output = model(images)

output=output[0]
output = output.argmax(0)

palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 19 - 1])
colors = torch.as_tensor([i for i in range(19)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output.byte().cpu().numpy())
r.putpalette(colors)

plt.imshow(r)
plt.show()


import torch
from datasets import load_dataset
from matplotlib.pyplot import plot
from pygments.formatters import img
from torchvision import transforms
from torchvision.transforms import v2

dataset = load_dataset("sfarrukhm/intel-image-classification")

transforms.ToTensor()

CLASSES = ["Building","Forest","Glacier","Mountain","Sea","Street"]

transforms = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
out = transforms(img)

plot([img, out])
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from src.config import IMG_SIZE,VAL_SPLIT, SEED,CLASSES,STRATIFY_SPLIT
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2

class inteldataset(Dataset):

    def __init__(self,hf_split, indices = None, transform = None):
        self.data = hf_split
        self.indices = list(range(len(hf_split))) if indices is None else list(indices)
        self.base = v2.Compose([v2.Resize(IMG_SIZE),v2.ToTensor()])
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self,i):
        ex = self.data[self.indices[i]]
        img = self.base(ex["image"].convert("RGB"))
        if self.transform is not None:
            img = self.transform(img)
        return img, ex["label"]

def load_datasets(val_size=VAL_SPLIT, seed=SEED, train_transform=None):

    ds = load_dataset("sfarrukhm/intel-image-classification")
    hf_train,hf_test =ds["train"],ds["test"]

    hf_names = hf_train.features["label"].names
    assert hf_names == CLASSES, f"Classes hf {hf_names} != config {CLASSES}"

    labels= hf_train["label"]
    train_idx, val_idx = train_test_split(
        list(range(len(hf_train))),
        test_size= val_size,
        stratify= labels if STRATIFY_SPLIT else None,
        random_state=seed
    )

    return inteldataset(hf_train,train_idx, transform = train_transform),inteldataset(hf_train,val_idx),inteldataset(hf_test)

def trainAugmentation(img_size: tuple[int,int] = IMG_SIZE) -> v2.Compose:
    """Return a transform for training data augmentation."""
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1), antialias=True),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.20, hue=0.02),
        v2.RandomErasing(p=0.5, scale=(0.02, 0.1)),
    ])
from torch.utils.data import Dataset
from torchvision import transforms
from datasets import load_dataset
from config import IMG_SIZE,VAL_SPLIT, SEED,CLASSES,STRATIFY_SPLIT
from sklearn.model_selection import train_test_split

class inteldataset(Dataset):

    def __init__(self,hf_split, indices = None, transform = None):
        self.data = hf_split
        self.indices = list(range(len(hf_split))) if indices is None else list(indices)
        self.base = transforms.Compose([transforms.Resize(IMG_SIZE),transforms.ToTensor()])
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
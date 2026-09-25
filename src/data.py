from torch.utils.data import dataset
from torchvision import transforms
from datasets import load_dataset
from src.config import IMG_SIZE,VAL_SPLIT, SEED,CLASSES,STRATIFY_SPLIT
from sklearn.model_selection import train_test_split

class dataset(dataset):

    def __init__(self,hf_split):
        self.data = hf_split
        self.indices = list(range(len(hf_split))) if indices is None else list(indices)
        self.base = transform.compose([transforms.Resize(IMG_SIZE),transforms.ToTensor()])
        self.transform = transforms
    
    def __len__(self):
        return len(self.indices)
    
    def __getItem__(self,i):
        ex = self.data[self.indices[i]]
        img = self.base(ex["image"].convert("RGB"))
        if self.transform is not None:
            img = self.transform(img)
        return img, ex["label"]

def load_datasets(val_size=VAL_SPLIT, seed=SEED, train_transform=None):

    ds = load_dataset("sfarrukhm/intel-image-classification")
    hf_train,hf_test =ds["train"],ds["test"]

    hf_names = hf_train.features["label"].names
    assert hf_names == CLASSES, f"Classes hf {names} != config {CLASSES}"

    labels= hf_train["label"]
    train_idx, val_idx = train_test_split(
        list(range(len(hf_train))),
        test_size= val_size,
        stratify= labels if STRATIFY_SPLIT else None,
        random_state=seed
    )

    return dataset(hf_train,train_idx, transform = train_transform),dataset(hf_train,val_idx),dataset(hf_test)


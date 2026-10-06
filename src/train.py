from config import DEVICE,NUM_CLASSES,SEED,IMG_SIZE,setSeed,seedWorker,getGenerator
from data import load_datasets
import copy 
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class BaselineCNN(nn.Module):

    def __init__(self, num_classes: int = NUM_CLASSES, dropout1: float = 0.5,
                 dropout2: float = 0.3):
        super().__init__()

        def block(c_in,c_out):
            return nn.Sequential(
                nn.Conv2d(c_in,c_out,kernel_size=3,padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(3,32),
            block(32,64),
            block(64,128),
            block(128,256)
        )

        flat_dim = 256 * (IMG_SIZE[0] // 16) * (IMG_SIZE[1] // 16)

        self.classifier= nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(inplace=True),
        nn.Dropout(dropout1),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout2),
           nn.Linear(128,num_classes)
        )

    def forward(self,x):
        return self.classifier(self.features(x))
    
def makeLoaders(train_ds,val_ds,test_ds,batch_size: int =64,  num_workers: int = 2, seed: int = SEED):
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seedWorker,
        pin_memory=DEVICE.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_ds,shuffle=True, generator=getGenerator(seed), **common)
    val_loarder = DataLoader(val_ds,shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loarder, test_loader

def train_one_epoch(model,loader,criterion,optimizer, device= DEVICE):
    model.train()
    total_loss, correct, n = 0.0,0,0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits,y)
        loss.backward()
        optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        n += bs
    return total_loss / n,correct/n

@torch.no_grad()
def validate(model, loader, criterion, device = DEVICE):
    model.eval()
    total_loss, correct, n = 0.0, 0 ,0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        bs = y.size(0)
        total_loss += criterion(logits, y).item() * bs
        correct += (logits.argmax(1) == y).sum().item()
        n += bs
    return total_loss/ n, correct/n

#entrainement 

def fit_model(model,train_loader, val_loader, epochs: int = 30, lr: float = 1e-3, weight_decay: float = 0.0, patience: int = 5,
              min_delta: float = 1e-4, device = DEVICE, verbose: bool = True):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer= torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_loss, best_epoch, wait = float("inf"), 0 ,0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, epochs + 1):
        t0 =  time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model,train_loader,criterion,optimizer,device)
        va_loss, va_acc = validate(model,val_loader,criterion,device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)

        if va_loss < best_loss - min_delta:
            best_loss,best_epoch,wait = va_loss,epoch,0
            best_state = copy.deepcopy(model.state_dict())
            flag = " * best"
        else:
            wait +=1
            flag = f"  (patience{wait}/{patience})"
        if verbose:
            print(f"epoch {epoch:>3}/{epochs} | "
                  f"train_loss {tr_loss:.4f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.3f} | "
                  f"{time.perf_counter() - t0:5.1f}s{flag}")
        if wait >= patience:
            if verbose:
                print(f"early stopping à epoch : {epoch}  best epoch : {best_epoch}")
            break
    model.load_state_dict(best_state)
    history["best_epoch"]= best_epoch
    history["best_val_loss"] = best_loss
    return model,history

if __name__== "__main__":
    setSeed()
    train_ds,val_ds,test_ds = load_datasets()
    train_loader, val_loader, test_loader = makeLoaders(train_ds,val_ds,test_ds)

    model= BaselineCNN()
    print("param : ", sum(p.numel() for p in model.parameters()))

    model, history = fit_model(model,train_loader,val_loader,epochs=30,lr=1e-3,patience=5)
    print(f"meilleur epoch : {history['best_epoch']} |"
          f"val loss {history['best_val_loss']:.4f} | "
          f"val acc {history['val_acc'][history['best_epoch'] - 1]:.3f}")
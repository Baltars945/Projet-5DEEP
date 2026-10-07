from src.config import DEVICE,SEED,setSeed,seedWorker,getGenerator, FIGURES_DIR, MODELS_DIR, CLASSES
from src.model_io import saveModel, checkRoundTrip
from src.evaluate import evaluateModel, metricsSummary, saveMetrics, plotHistory, pltConfusionMatrix
from src.data import load_datasets, trainAugmentation
from src.models import BaselineCNN, buildBaseline, summarizeModel
from src.tuning import gridSearch, searchResults, saveSearch, splitBestParams
import copy 
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


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

    # Recherche sur des donnees NON augmentees : gridSearch ne fait que decouper
    # des indices sur un seul dataset, donc le fold de validation de chaque split
    # heriterait de l'augmentation et fausserait le classement des combinaisons.
    train_ds,val_ds,test_ds = load_datasets()

    search = gridSearch(train_ds, cv=3, max_epochs=5, subset=2000)
    print(f"meilleur score CV ({search.scoring}) : {search.best_score_:.4f}")
    print(f"meilleurs parametres : {search.best_params_}")
    for row in searchResults(search):
        print(f"  rank {row['rank']:>2} | {row['mean_test_score']:.4f} +/- {row['std_test_score']:.4f}")
    saveSearch(search, MODELS_DIR / "gridsearch_results.json")

    module_params, fit_params = splitBestParams(search.best_params_)
    name = "baseline_cnn"

    # Augmentation activee pour le re-entrainement final uniquement (split train).
    # Le dataset HuggingFace est en cache : ce second appel ne retelecharge rien.
    train_ds,val_ds,test_ds = load_datasets(train_transform=trainAugmentation())
    train_loader, val_loader, test_loader = makeLoaders(train_ds,val_ds,test_ds)

    model = BaselineCNN(**module_params)
    print(summarizeModel(model))

    model, history = fit_model(model,train_loader,val_loader,epochs=30,lr=fit_params['lr'],patience=5)
    path = saveModel(model,name)
    print(f"model saved to {path}")

    metrics = evaluateModel(model,test_loader)
    print(metricsSummary(name,metrics))

    saveMetrics(metrics,MODELS_DIR/f"{name}_metrics.json")

    plotHistory(history,title=name,save_path=FIGURES_DIR/f"{name}_history.png")
    pltConfusionMatrix(metrics["confusion_matrix"],CLASSES,title=name,save_path=FIGURES_DIR/f"{name}_confusion_matrix.png")

    sample, _ = test_ds[0]
    checkRoundTrip(model,buildBaseline,name,sample.unsqueeze(0))
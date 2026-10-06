from data import inteldataset,load_datasets
import numpy as np
from config import (GRAPH_DIR,CLASSES,setSeed,NUM_CLASSES,SEED,IMG_SIZE,SOURCE_IMG_SIZE
                    ,IMAGENET_MEAN)
import matplotlib.pyplot as plt

def labels(ds: inteldataset) -> np.ndarray:
    all_labels = np.asarray(list(ds.data["label"]))
    return all_labels[ds.indices]

def raw_image(ds: inteldataset, i: int):
    return ds.data[ds.indices[i]]["image"]

def save_image(fig, name: str, save: bool) -> None:
    if save:
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(GRAPH_DIR / f"{name}.png", dpi=150, bbox_inches="tight")

def describe_dataset(train: inteldataset, val: inteldataset, test: inteldataset) -> None:
    total = len(train) + len(val) + len(test)
    print("=" * 60)
    print("dataset : intel image classification")
    print("=" * 60)
    print(f"Classes : {CLASSES}")
    print(f"Nb image : {total}")
    for name, ds in (("train", train), ("val",val),("test",test)):
        print(f"   {name:<5}: {len(ds):>6} images ({100 * len(ds) / total:5.1f}%)")

    raw = raw_image(train,0)
    print("\nImg avant prétraitement")
    print(f" type    : {type(raw).__name__}")
    print(f" format  : {raw.format}")
    print(f" mode    : {raw.mode}")
    print(f" taille  : {raw.size[0]}x{raw.size[1]}")
    
    x,y = train[0]
    print("\n img post prétraitement")
    print(f" shape   : {tuple(x.shape)}  (C,H,W)")
    print(f" dtype   : {x.dtype}")
    print(f" valeurs : [{x.min():.3f}, {x.max():.3f}]")
    print(f" label   : {y} -> {CLASSES[y]}")

def plot_class_distribution(train,val,test,save: bool = True):
    splits = {"train" : train,"val": val, "test" : test}
    counts = {k: np.bincount(labels(ds),minlength=NUM_CLASSES)for k, ds in splits.items()}

    x = np.arange(NUM_CLASSES)
    w = 0.27
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,5))
    for j, (name, c) in enumerate(counts.items()):
        bars = ax1.bar(x+(j-1)*w,c,w,label=f"{name}(n={c.sum()})")
        ax1.bar_label(bars,fontsize = 7, padding=2)
        ax2.bar(x+(j-1)* w,100*c/c.sum(),w,label=name)
    ax1.set_title("Nb images par classes")
    ax1.set_ylabel("images")
    ax2.set_title("proportion par classe")
    ax2.set_label("%")
    for ax in (ax1,ax2):
        ax.set_xticks(x,CLASSES,rotation=30)
        ax.legend()
        ax.grid(axis="y",alpha=0.3)
    fig.suptitle("distribution des classes",fontsize=14)
    fig.tight_layout()
    save_image(fig,"class_distribution",save)

    for name, c in counts.items():
        print(f"{name:<5}: min={c.min()}({CLASSES[c.argmin()]}),"
              f"max={c.max()}({CLASSES[c.argmax()]}), ration max/min={c.max()/ c.min():.2f}")
    return fig

def plot_examples_per_class(ds:inteldataset,n_per_class: int = 6,seed: int = SEED, save: bool =True):
    rng = np.random.default_rng(seed)
    label = labels(ds)
    fig,axes = plt.subplots(NUM_CLASSES,n_per_class,figsize=(2*n_per_class,2.1*NUM_CLASSES))
    for c, cls in enumerate(CLASSES):
        pool = np.flatnonzero(label == c)
        chosen = rng.choice(pool,size=min(n_per_class,len(pool)),replace=False)
        for j,ax in enumerate(axes[c]):
            ax.set_xticks([])
            ax.set_yticks([])
            if j < len(chosen):
                img = raw_image(ds,int(chosen[j]))
                ax.imshow(img)
                ax.set_title(f"{img.size[0]}x{img.size[1]}",fontsize=7)
            if j == 0:
                ax.set_ylabel(cls, fontsize=11,rotation=0, ha="right",va="center")
    fig.suptitle("Exemples par classes",fontsize=14)
    fig.tight_layout()
    save_image(fig,"examples_per_class",save)
    return fig

def get_image_sizes(ds:inteldataset,max_samples:int | None = None, seed: int =SEED) -> np.ndarray:
    idx = np.arange(len(ds))
    if max_samples is not None and max_samples<len(ds):
        idx = np.random.default_rng(seed).choice(idx,size=max_samples,replace=False)
    return np.array([raw_image(ds,int(i)).size for i in idx])

def plot_image_dimensions(train,val,test,max_samples:int|None=None,save:bool = True):
    splits={"train":train,"val":val,"test":test}
    sizes = {k:get_image_sizes(ds,max_samples)for k,ds in splits.items()}
    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,5))

    for name, s in sizes.items():
        uniq,cnt=np.unique(s,axis=0,return_counts=True)
        ax1.scatter(uniq[:,0],uniq[:,1],s=20+200*cnt/cnt.max(),alpha=0.6,label=name)
    ax1.set_xlabel("largeur (px)")
    ax1.set_ylabel("hauteur (px)")
    ax1.set_title("dimensions brutes (taille du point ∝ nombre d'images)")
    ax1.legend()
    ax1.grid(alpha=0.3)

    std=SOURCE_IMG_SIZE
    names= list(sizes)
    n_std=[int(np.all(s==std,axis=1).sum())for s in sizes.values()]
    n_other=[len(s)- a for s, a in zip(sizes.values(),n_std)]
    b1 = ax2.bar(names,n_std,label=f"{std[0]}x{std[1]}")
    b2 = ax2.bar(names,n_other,bottom=n_std,label="autre taille")
    ax2.bar_label(b2,labels=[str(o)for o in n_other], fontsize=9)
    ax2.set_title("image taille standard vs non standard")
    ax2.set_ylabel("image")
    ax2.legend()
    ax2.grid(axis="y",alpha=0.3)

    fig.suptitle(f"dimensions des images (redimensionnés en {IMG_SIZE[0]}x{IMG_SIZE[1]})", fontsize=13)
    fig.tight_layout()
    save_image(fig,"image_dimensions",save)

def plot_color_stats(ds: inteldataset , max_per_class: int = 300, seed: int = SEED, save: bool = True):
    rng = np.random.default_rng(seed)
    label = labels(ds)
    means = np.zeros((NUM_CLASSES,3))
    hist = np.zeros((3,50))
    bins = np.linspace(0,1,51)

    for c in range(NUM_CLASSES):
        pool = np.flatnonzero(label == c)
        chosen = rng.choice(pool, size = min(max_per_class, len(pool)), replace=False)
        acc = []
        for i in chosen:
            x = ds.base(raw_image(ds, int(i)).convert("RGB")).numpy()
            acc.append(x.mean(axis=(1,2)))
            for ch in range(3):
                hist[ch] += np.histogram(x[ch], bins=bins)[0]
            means[c] = np.mean(acc, axis= 0)
    colors = ["tab:red","tab:green","tab:blue"]
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(15,5))
    x = np.arange(NUM_CLASSES)
    w = 0.27
    for ch, (col,name) in enumerate(zip(colors,"RGB")):
        ax1.bar(x + (ch -1) * w,means[:, ch], w, color= col, alpha=0.8, label = name)
        ax1.axhline(IMAGENET_MEAN[ch],color = col, ls="--", lw=1)
    ax1.set_xticks(x,CLASSES, rotation=30)
    ax1.set_ylabel("intensité mayenne [0,1]")
    ax1.set_title("moyenne par canal et par classe")
    ax1.legend()
    ax1.grid(axis="y",alpha=0.3)
    
    centers = (bins[:-1] + bins[1:]) / 2
    for ch, (col, name) in enumerate(zip(colors, "RGB")):
        ax2.plot(centers,hist[ch]/ hist[ch].sum(), color = col, label=name)
    ax2.set_xlabel("intensité du pixel")
    ax2.set_ylabel("frequence")
    ax2.set_title("distribution des intensités par canal")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("statistique de couleur",fontsize=4)
    fig.tight_layout()
    save_image(fig,"color_stats",save)
    return fig
#launcher

def explore_datasets(train,val, test,    save:bool=  True,max_size_sampe:int | None = None) -> None:
    describe_dataset(train,val,test)
    plot_class_distribution(train,val,test,save=save)
    plot_examples_per_class(train,save=save)
    plot_image_dimensions(train,val,test,max_size_sampe,save=save)
    plot_color_stats(train,save=save)
    plt.show()
    
if __name__ == "__main__":
    setSeed()
    train_ds,val_ds,test_ds = load_datasets()
    explore_datasets(train_ds,val_ds,test_ds)
    
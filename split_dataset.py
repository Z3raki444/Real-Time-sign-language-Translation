import os, shutil, random, pathlib
SRC = pathlib.Path("Data")
OUT = pathlib.Path("DataSplit")
random.seed(42)

for subset in ["Train","Val","Test"]:
    (OUT/subset).mkdir(parents=True, exist_ok=True)

ratio = (0.7, 0.15, 0.15)  # Train/Val/Test

classes = [d.name for d in SRC.iterdir() if d.is_dir()]
for cls in classes:
    files = sorted((SRC/cls).glob("*.jpg"))
    random.shuffle(files)
    n = len(files)
    n_tr, n_va = int(n*ratio[0]), int(n*(ratio[0]+ratio[1]))
    splits = {"Train": files[:n_tr], "Val": files[n_tr:n_va], "Test": files[n_va:]}
    for subset, items in splits.items():
        dst = OUT/subset/cls
        dst.mkdir(parents=True, exist_ok=True)
        for f in items:
            shutil.copy2(f, dst/f.name)

print("Done. Classes:", classes)

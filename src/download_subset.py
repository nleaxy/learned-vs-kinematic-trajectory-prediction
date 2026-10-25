import random
import subprocess
import sys
from pathlib import Path

N_SCENES = 60000
SEED = 42
BUCKET = "s3://argoverse/datasets/av2/motion-forecasting/train"

s5cmd = sys.argv[1]

# list all scenario folders in train, id is the first part of each path
out = subprocess.run(
    [s5cmd, "--no-sign-request", "ls", f"{BUCKET}/"],
    capture_output=True, text=True, check=True,
).stdout
ids = sorted({line.split()[-1].split("/")[0] for line in out.splitlines() if line.strip()})
ids = [i for i in ids if len(i) == 36]
print(f"found {len(ids)} scenarios")

chosen = sorted(random.Random(SEED).sample(ids, N_SCENES))

Path("configs").mkdir(exist_ok=True)
Path("configs/train_subset.txt").write_text("\n".join(chosen) + "\n")

# one copy command per scenario, maps are skipped
with open("download_train.txt", "w") as f:
    for sid in chosen:
        f.write(f'cp "{BUCKET}/{sid}/scenario_{sid}.parquet" data/train/{sid}/\n')
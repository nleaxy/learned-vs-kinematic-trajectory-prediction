import numpy as np
import pandas as pd

from groups import LAST_OBS, road_user_group
from kinematic import T, cv, errors

z = np.load("data/processed/val.npz")
xy = z["xy"]
vehicle = road_user_group(z["object_type"]) == "vehicle"

# velocity from the last two points as in our CV, and mean velocity over the whole history as in the Argoverse 1 baseline
last = cv(xy, 1)
v = (xy[:, LAST_OBS] - xy[:, 0]) / (LAST_OBS * 0.1)
mean = xy[:, LAST_OBS][:, None] + v[:, None] * T[None, :, None]

rows = []
for name, pred in [("last two points", last), ("mean over history", mean)]:
    ade, fde = errors(pred, xy)
    rows.append({"velocity": name, "n_vehicles": int(vehicle.sum()), "ade": ade[vehicle].mean(), "fde": fde[vehicle].mean()})
    print(f"CV, {name}: vehicles on val, ADE {ade[vehicle].mean():.2f} m, FDE {fde[vehicle].mean():.2f} m")

pd.DataFrame(rows).round(3).to_csv("results/cv_mean_velocity.csv", index=False)
print("saved results/cv_mean_velocity.csv")
import numpy as np

LAST_OBS = 49
GROUPS = {
    "vehicle": "vehicle", "bus": "vehicle",
    "pedestrian": "pedestrian",
    "cyclist": "two_wheeler", "motorcyclist": "two_wheeler",
}


def road_user_group(types):
    return np.array([GROUPS.get(t, "other") for t in types])


def heading_change(xy, heading, min_dist=2.0):
    # start direction is the dataset heading at t 0, end direction is the last stretch of at least min_dist meters
    if not np.isfinite(heading).all():
        raise ValueError("heading has non finite values")
    fut = xy[:, LAST_OBS:]
    end = fut[:, -1]
    far = np.linalg.norm(fut - end[:, None], axis=2) >= min_dist
    last = far.shape[1] - 1 - np.argmax(far[:, ::-1], axis=1)
    last[~far.any(1)] = 0
    v = end - fut[np.arange(len(fut)), last]
    diff = np.arctan2(v[:, 1], v[:, 0]) - heading
    dpsi = np.degrees(np.abs((diff + np.pi) % (2 * np.pi) - np.pi))
    return dpsi, np.linalg.norm(v, axis=1)


def motion_type(xy, heading, straight=15, turn=45, min_path=1.0, min_dist=2.0):
    path = np.linalg.norm(np.diff(xy[:, LAST_OBS:], axis=1), axis=2).sum(1)
    dpsi, stretch = heading_change(xy, heading, min_dist)
    out = np.where(dpsi < straight, "straight", np.where(dpsi > turn, "turn", "intermediate")).astype(object)
    # only jitter around one spot, no direction at the end
    out[stretch < min_dist] = "undefined"
    out[path < min_path] = "stationary"
    return out.astype(str), dpsi
import random
import numpy as np


# EventDrop augmentation by randomly dropping events
def event_drop(events, resolution):
    raw_events = events
    option = np.random.randint(
        0, 4
    )  # 0: identity, 1: drop_by_time, 2: drop_by_area, 3: random_drop
    if option == 0:  # identity, do nothing
        return events
    elif option == 1:  # drop_by_time
        T = np.random.randint(1, 10) / 10.0  # np.random.uniform(0.1, 0.9)
        events = drop_by_time(events, T=T)
    elif option == 2:  # drop by area
        area_ratio = (
            np.random.randint(1, 20) / 20.0
        )  # np.random.uniform(0.05, 0.1, 0.15, 0.2, 0.25)
        events = drop_by_area(events, resolution=resolution, area_ratio=area_ratio)
    elif option == 3:  # random drop
        ratio = np.random.randint(1, 10) / 10.0  # np.random.uniform(0.1, 0.9)
        events = random_drop(events, ratio=ratio)

    if len(events) == 0:  # avoid dropping all the events
        events = raw_events

    return events


# drop the events within a given time interval
# param: events contains x, y, t, p information
def drop_by_time(events, T=0):
    # assert 0.1 <= T <= 0.5

    # time interval
    t_start = np.random.uniform(0, 1)
    if T == 0:  # randomly choose a value between [0.1, 0.9]
        T = np.random.randint(1, 10) / 10.0
    t_end = t_start + T

    timestamps = events[:, 0]
    max_t = max(timestamps)
    idx = (timestamps < (max_t * t_start)) | (
        timestamps > (max_t * t_end)
    )  # remaining events that are not within the given time interval

    return events[idx]


# drop the events within a a fixed area constrained by X, Y
# area ratio: the ratio between the area whose pixels are dropped and the overall pixel range
def drop_by_area(events, resolution=(128, 128), area_ratio=0):
    # assert 0.1 <= area_ratio <= 0.3

    # get the area whose events are to be dropped
    x0 = np.random.uniform(resolution[0])
    y0 = np.random.uniform(resolution[1])

    if area_ratio == 0:
        area_ratio = np.random.randint(1, 6) / 20.0

    x_out = resolution[0] * area_ratio
    y_out = resolution[1] * area_ratio

    x0 = int(max(0, x0 - x_out / 2.0))
    y0 = int(max(0, y0 - y_out / 2.0))

    x1 = min(resolution[0], x0 + x_out)
    y1 = min(resolution[1], y0 + y_out)

    xy = (x0, x1, y0, y1)  # rectangele to be dropped

    idx1 = (events[:, 2] < xy[0]) | (events[:, 2] > xy[1])
    idx2 = (events[:, 3] < xy[2]) | (events[:, 3] > xy[3])
    idx = idx1 & idx2

    return events[idx]


# randomly drop a proportion of events
def random_drop(events, ratio=0):
    # assert 0.1 <= ratio <= 0.5

    if ratio == 0:
        ratio = np.random.randint(1, 10) / 10.0

    N = events.shape[0]  # number of total events
    num_drop = int(N * ratio)  # number of events to be dropped
    idx = random.sample(list(np.arange(0, N)), N - num_drop)

    return events[idx]


# randomly shift events
def random_shift_events(events, max_shift=20, resolution=(128, 128), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=(2,))
        events[:, 2] += x_shift
        events[:, 3] += y_shift
        valid_events = (
            (events[:, 2] >= 0)
            & (events[:, 2] < W)
            & (events[:, 3] >= 0)
            & (events[:, 3] < H)
        )
        events = events[valid_events]
    return events


# randomly flip events
def random_flip_events_along_x(events, resolution=(128, 128), p=0.5):
    H, W = resolution
    if np.random.random() < p:
        events[:, 2] = W - 1 - events[:, 2]
    return events

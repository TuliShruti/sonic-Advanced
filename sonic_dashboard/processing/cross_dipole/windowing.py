def get_windows(gather, window_length, step):
    windows = []
    indices = []

    for t in range(0, gather.shape[1] - window_length, step):
        windows.append(gather[:, t:t+window_length])
        indices.append(t)

    return windows, indices

import numpy as np

def build_background(X_single, n_samples=30, noise_std=0.06):
    background = []

    for _ in range(n_samples):
        noise = np.random.normal(0, noise_std, size=X_single.shape)
        x = np.clip(X_single + noise, 0, 1)
        background.append(x)

    return np.vstack(background)

import cv2
import numpy as np

FEATURE_NAMES = [
    "fft_peak_ratio",
    "fft_high_freq_energy",
    "fft_spectral_flatness",
    "fft_peak_strength",
    "lbp_entropy",
    "highlight_clip_ratio",
    "glare_blob_ratio",
    "saturation_mean",
    "saturation_std",
    "sharpness_uniformity",
    "sharpness_mean",
    "edge_line_density",
    "rgb_channel_mismatch",
]


def _resize(img, max_side=512):
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def fft_features(gray):
    f = np.fft.fft2(gray.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.log1p(np.abs(fshift))

    h, w = mag.shape
    cy, cx = h // 2, w // 2

    r_low = max(2, min(h, w) // 20)
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    mid_mask = (dist > r_low) & (dist <= min(h, w) // 2)

    mid_vals = mag[mid_mask]
    total_energy = mag.sum() + 1e-8

    mean_mid = mid_vals.mean()
    std_mid = mid_vals.std()
    peak_ratio = (mid_vals.max() - mean_mid) / (std_mid + 1e-6)
    high_freq_energy = mid_vals.sum() / total_energy
    spectral_flatness = np.exp(np.mean(np.log(mid_vals + 1e-8))) / (mid_vals.mean() + 1e-8)

    K = min(8, mid_vals.size)
    top_sum = mid_vals[np.argpartition(mid_vals, -K)[-K:]].sum()
    peak_strength = top_sum / (mid_vals.sum() + 1e-8)

    return float(peak_ratio), float(high_freq_energy), float(spectral_flatness), float(peak_strength)


def lbp_entropy(gray):
    gray = gray.astype(np.int16)
    h, w = gray.shape
    center = gray[1:-1, 1:-1]
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1)]
    code = np.zeros_like(center, dtype=np.uint8)
    for i, (dy, dx) in enumerate(offsets):
        neighbour = gray[1 + dy:1 + dy + center.shape[0],
                          1 + dx:1 + dx + center.shape[1]]
        code |= ((neighbour >= center).astype(np.uint8) << i)

    hist, _ = np.histogram(code, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist /= (hist.sum() + 1e-8)
    hist = hist[hist > 0]
    entropy = -(hist * np.log2(hist)).sum()
    return float(entropy)


def highlight_clip_ratio(img):
    bright = (img >= 245).all(axis=2)
    return float(bright.mean())


def saturation_stats(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].astype(np.float32) / 255.0
    return float(s.mean()), float(s.std())


def glare_blob_ratio(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    bright = (v >= 245).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(bright)
    if n_labels <= 1:
        return 0.0
    area = img.shape[0] * img.shape[1]
    blob_areas = [(labels == lab).sum() for lab in range(1, n_labels)]
    large_area = sum(a for a in blob_areas if a >= area * 0.005)
    return float(large_area / area)


def rgb_channel_mismatch(img):
    chans = cv2.split(img)
    mags = []
    for c in chans:
        gx = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)
        mags.append(np.sqrt(gx * gx + gy * gy) + 1e-8)
    diff_rg = np.mean(np.abs(mags[2] - mags[1]))
    diff_gb = np.mean(np.abs(mags[1] - mags[0]))
    diff_rb = np.mean(np.abs(mags[2] - mags[0]))
    mean_mag = np.mean(mags)
    return float((diff_rg + diff_gb + diff_rb) / (3 * mean_mag + 1e-8))


def sharpness_grid(gray, grid=4):
    h, w = gray.shape
    cell_h, cell_w = h // grid, w // grid
    vals = []
    for i in range(grid):
        for j in range(grid):
            cell = gray[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w]
            if cell.size == 0:
                continue
            lap = cv2.Laplacian(cell, cv2.CV_64F)
            vals.append(lap.var())
    vals = np.array(vals)
    return float(vals.std()), float(vals.mean())


def edge_line_density(gray):
    edges = cv2.Canny(gray, 80, 160)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                             minLineLength=min(gray.shape) // 3, maxLineGap=10)
    n_lines = 0 if lines is None else len(lines)
    area = gray.shape[0] * gray.shape[1]
    return float(n_lines) / (area / 1e5)


def extract_features(img_path_or_array):
    if isinstance(img_path_or_array, str):
        img = cv2.imread(img_path_or_array)
        if img is None:
            raise ValueError(f"Could not read image: {img_path_or_array}")
    else:
        img = img_path_or_array

    img = _resize(img, 512)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    peak_ratio, hf_energy, fft_flatness, fft_peak_strength = fft_features(gray)
    lbp_e = lbp_entropy(gray)
    clip_ratio = highlight_clip_ratio(img)
    glare_ratio = glare_blob_ratio(img)
    sat_mean, sat_std = saturation_stats(img)
    sharp_std, sharp_mean = sharpness_grid(gray)
    line_density = edge_line_density(gray)
    rgb_mismatch = rgb_channel_mismatch(img)

    feats = np.array([
        peak_ratio,
        hf_energy,
        fft_flatness,
        fft_peak_strength,
        lbp_e,
        clip_ratio,
        glare_ratio,
        sat_mean,
        sat_std,
        sharp_std,
        sharp_mean,
        line_density,
        rgb_mismatch,
    ], dtype=np.float64)

    return feats

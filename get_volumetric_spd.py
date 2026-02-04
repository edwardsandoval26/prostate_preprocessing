import os
import json
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
import warnings
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tqdm import tqdm
from time import sleep

# -------------------------------
# Silence warnings & TF logs
# -------------------------------
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")

# -------------------------------
# Utils
# -------------------------------
def compute_covariance(matrix):
    if matrix.shape[1] < 2:
        return np.zeros((matrix.shape[0], matrix.shape[0]))
    centered = matrix - np.mean(matrix, axis=1, keepdims=True)
    return np.matmul(centered, centered.T) / (matrix.shape[1] - 1)

def get_spd(cov_matrix, epsilon=1e-5):
    trace = np.trace(cov_matrix)
    return cov_matrix + (np.eye(cov_matrix.shape[0]) * epsilon * (trace if trace > 0 else 1.0))

def compute_sextant_size(z_min, z_max, y_min, y_max, x_min, x_max):
    dz = z_max - z_min
    dy = y_max - y_min
    dx = x_max - x_min

    # 2 (L/R) × 3 (base/mid/apex)
    sz = max(1, int(round(dz / 3)))
    sy = max(1, int(round(dy)))
    sx = max(1, int(round(dx / 2)))

    return sz, sy, sx

# -------------------------------
# CONFIG
# -------------------------------
PATH_IMAGES = '/Datasets/PICAI_olmos/Task2308_prep_picai_baseline/imagesTr'
PATH_SEG = '/Datasets/PICAI_olmos/Task2308_prep_picai_baseline/labelsTr'
PATH_GLAND = '/Datasets/PICAI_olmos/Task2308_prep_picai_baseline/labelsTr_bosma'
PATH_JSON_INFO = '/Datasets/PICAI_olmos/info-12x32x32.json'
SAVE_PATH = '/Datasets/PICAI_32x32x12/volumetric_own_spd'
os.makedirs(SAVE_PATH, exist_ok=True)

print("Loading VGG19 Model...")
vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(None, None, 3))
feature_extractor = Model(inputs=vgg19.input,
                          outputs=vgg19.get_layer('block1_conv2').output)

with open(PATH_JSON_INFO, 'r') as f:
    info = json.load(f)

# -------------------------------
# Main loop
# -------------------------------
for patient_id in tqdm(info, desc="Processing patients", unit="vol"):

    t2 = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(PATH_IMAGES, f'{patient_id}_0000.nii.gz'))
    )
    adc = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(PATH_IMAGES, f'{patient_id}_0001.nii.gz'))
    )
    bval = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(PATH_IMAGES, f'{patient_id}_0002.nii.gz'))
    )

    gland_mask = sitk.GetArrayFromImage(
        sitk.ReadImage(os.path.join(PATH_GLAND, f'{patient_id}.nii.gz'))
    )

    coords = np.argwhere(gland_mask > 0)
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0) + 1

    # --- Standardize ---
    t2 = (t2 - t2.mean()) / (t2.std() + 1e-8)
    adc = (adc - adc.mean()) / (adc.std() + 1e-8)
    bval = (bval - bval.mean()) / (bval.std() + 1e-8)

    # --- Sextant VROI ---
    sz, sy, sx = compute_sextant_size(z_min, z_max, y_min, y_max, x_min, x_max)

    cz, cy, cx = info[patient_id]['centroid']

    z_start = max(0, int(cz - sz // 2))
    z_end   = min(t2.shape[0], z_start + sz)

    y_start = max(0, int(cy - sy // 2))
    y_end   = min(t2.shape[1], y_start + sy)

    x_start = max(0, int(cx - sx // 2))
    x_end   = min(t2.shape[2], x_start + sx)

    # recentre if clipped
    z_start = max(0, z_end - sz)
    y_start = max(0, y_end - sy)
    x_start = max(0, x_end - sx)

    t2_crop   = t2[z_start:z_end, y_start:y_end, x_start:x_end]
    adc_crop  = adc[z_start:z_end, y_start:y_end, x_start:x_end]
    bval_crop = bval[z_start:z_end, y_start:y_end, x_start:x_end]

    features_collector = []
    depth, h, w = t2_crop.shape

    if depth > 0 and h > 0 and w > 0:
        for z in range(depth):
            batch_in = np.stack([
                np.repeat(t2_crop[z][:, :, None], 3, axis=2),
                np.repeat(adc_crop[z][:, :, None], 3, axis=2),
                np.repeat(bval_crop[z][:, :, None], 3, axis=2)
            ], axis=0)

            activations = feature_extractor.predict(batch_in, verbose=0)
            features_collector.append(activations)

        full_feats = np.stack(features_collector, axis=0)
        full_feats = full_feats.transpose(1, 4, 0, 2, 3)
        features_flat = full_feats.reshape(3 * 64, -1)

        cov_matrix = compute_covariance(features_flat)
        spd_matrix = get_spd(cov_matrix)
    else:
        spd_matrix = np.eye(192)

    np.save(os.path.join(SAVE_PATH, f"{patient_id}_cov.npy"), spd_matrix)
    sleep(0.01)

print("✅ Done.")

import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array
def read_rgb_image_cv2(image_path):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb

def preprocess_for_classification_from_path(image_path, target_size):
    img = load_img(image_path, target_size=target_size, color_mode="rgb")
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img, img_array

def preprocess_for_segmentation(image_rgb, target_size):
    img = cv2.resize(image_rgb, target_size, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def clean_mask(mask_prob, threshold=0.5, min_area=50):
    mask = (mask_prob > threshold).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    cleaned = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            cleaned[labels == i] = 1

    return cleaned

def resize_mask_to_original(mask_small, original_shape):
    h, w = original_shape[:2]
    mask = cv2.resize(mask_small.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return mask

def create_overlay(image_rgb, binary_mask, alpha=0.35):
    overlay = image_rgb.copy()
    green_layer = np.zeros_like(image_rgb)
    green_layer[..., 1] = 255

    tumor_pixels = binary_mask.astype(bool)
    overlay[tumor_pixels] = (
        (1 - alpha) * overlay[tumor_pixels] + alpha * green_layer[tumor_pixels]
    ).astype(np.uint8)

    return overlay

def draw_bbox_and_label(image_rgb, binary_mask, class_name, confidence):
    output = image_rgb.copy()
    ys, xs = np.where(binary_mask > 0)

    label = f"{class_name} | {confidence:.2%}"

    if len(xs) > 0 and len(ys) > 0:
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())

        cv2.rectangle(output, (x1, y1), (x2, y2), (255, 0, 0), 3)

        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        text_x = x1
        text_y = max(y1 - 10, th + 12)

        cv2.rectangle(
            output,
            (text_x, text_y - th - baseline - 6),
            (text_x + tw + 8, text_y + 4),
            (255, 0, 0),
            -1
        )

        cv2.putText(
            output,
            label,
            (text_x + 4, text_y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
    else:
        cv2.putText(
            output,
            f"{label} | no tumor region found",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )

    return output

def draw_class_only_label(image_rgb, class_name, confidence):
    output = image_rgb.copy()
    label = f"{class_name} | {confidence:.2%}"

    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

    x, y = 15, 35

    cv2.rectangle(
        output,
        (x - 5, y - th - baseline - 5),
        (x + tw + 5, y + 5),
        (255, 0, 0),
        -1
    )

    cv2.putText(
        output,
        label,
        (x, y - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    return output

def save_rgb_image(path, image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)

def save_mask_image(path, binary_mask):
    cv2.imwrite(str(path), (binary_mask * 255).astype(np.uint8))
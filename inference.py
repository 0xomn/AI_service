import numpy as np
from utils import (
    read_rgb_image_cv2,
    preprocess_for_classification_from_path,
    preprocess_for_segmentation
)

def predict_combined(image_path, classification_model, segmentation_model, class_names):

    image_rgb = read_rgb_image_cv2(image_path)

    cls_input = preprocess_for_classification_from_path(image_path, (224,224))[1]

    cls_probs = classification_model.predict(cls_input, verbose=0)[0]
    pred_idx = int(np.argmax(cls_probs))
    pred_class = class_names[pred_idx]
    confidence = float(cls_probs[pred_idx])

    if pred_class == "no_tumor":
        seg_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)

    else:
        seg_input = preprocess_for_segmentation(image_rgb, (256,256))
        seg_prob = segmentation_model.predict(seg_input, verbose=0)[0,:,:,0]
        seg_mask = (seg_prob > 0.5).astype(np.uint8)

    return {
        "class": pred_class,
        "confidence": confidence
    }
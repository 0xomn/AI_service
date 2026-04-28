import tensorflow as tf

# ===== losses =====
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

bce = tf.keras.losses.BinaryCrossentropy()

def bce_dice_loss(y_true, y_pred):
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)

# ===== models =====
classification_model = tf.keras.models.load_model(
    "models/classification.keras",
    compile=False
)

segmentation_model = tf.keras.models.load_model(
    "models/segmentation.keras",
    custom_objects={
        "dice_coef": dice_coef,
        "dice_loss": dice_loss,
        "bce_dice_loss": bce_dice_loss
    },
    compile=False
)

CLASS_NAMES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']
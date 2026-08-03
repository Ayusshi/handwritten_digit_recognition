import cv2
import numpy as np


def preprocess_image(image_path):
    """
    Convert a custom handwritten digit image into
    MNIST-like 28x28 format.
    """

    # -----------------------------
    # Read image
    # -----------------------------
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # -----------------------------
    # Convert to grayscale
    # -----------------------------
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # Binary inverse threshold
    # -----------------------------
    _, thresh = cv2.threshold(
        gray,
        128,
        255,
        cv2.THRESH_BINARY_INV
    )

    # -----------------------------
    # Find contours
    # -----------------------------
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        raise ValueError("No digit found.")

    # -----------------------------
    # Largest contour
    # -----------------------------
    contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(contour)

    digit = thresh[y:y+h, x:x+w]

    # -----------------------------
    # Padding
    # -----------------------------
    digit = cv2.copyMakeBorder(
        digit,
        20,
        20,
        20,
        20,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # -----------------------------
    # Preserve aspect ratio
    # -----------------------------
    h, w = digit.shape

    max_dim = max(h, w)

    scale = 20 / max_dim

    new_w = int(w * scale)
    new_h = int(h * scale)

    digit = cv2.resize(
        digit,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # -----------------------------
    # Create 28x28 canvas
    # -----------------------------
    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = digit

    # -----------------------------
    # Center digit using image moments
    # -----------------------------
    M = cv2.moments(canvas)

    if M["m00"] != 0:

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        shift_x = int(14 - cx)
        shift_y = int(14 - cy)

        translation_matrix = np.float32([
            [1, 0, shift_x],
            [0, 1, shift_y]
        ])

        canvas = cv2.warpAffine(
            canvas,
            translation_matrix,
            (28, 28)
        )

    # -----------------------------
    # Save processed image (Optional)
    # -----------------------------
    cv2.imwrite("outputs/processed.png", canvas)

    return canvas
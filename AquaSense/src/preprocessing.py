"""
AquaSense — Physics-Informed Acoustic Preprocessing Engine

USP 1: Cleans noisy, low-visibility underwater sonar imagery using
Non-Local Means Denoising (speckle reduction) and Contrast Limited
Adaptive Histogram Equalization (CLAHE) prior to neural inference.

This module is designed for real-world sonar/acoustic camera feeds
where speckle noise and uneven illumination degrade detection accuracy.
"""

import cv2
import numpy as np


def clean_sonar_frame(img_np):
    """
    Applies acoustic speckle filtering and CLAHE to raw sonar images.

    Pipeline:
        1. Convert to grayscale (if color input)
        2. Non-Local Means Denoising — suppresses multiplicative speckle noise
           common in side-scan sonar and acoustic camera imagery
        3. CLAHE — equalizes contrast across uneven underwater illumination
           gradients without over-amplifying noise

    Args:
        img_np (np.ndarray): Input image array (RGB, BGR, or Grayscale).

    Returns:
        np.ndarray: Enhanced single-channel grayscale image (uint8).
    """
    if img_np is None:
        raise ValueError("Input image is None.")

    # Convert to grayscale if multi-channel
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np.copy()

    # 1. Speckle Noise Reduction Filter
    #    h=10 controls filter strength; higher = more smoothing
    #    templateWindowSize=7, searchWindowSize=21 are standard for sonar
    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=10,
        templateWindowSize=7,
        searchWindowSize=21
    )

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    #    clipLimit=2.5 prevents over-amplification of noise
    #    tileGridSize=(8,8) provides localized equalization
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    return enhanced

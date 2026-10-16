import cv2
import numpy as np


class ShadowFeatureExtractor:
    """
    Extracts simple shadow/contrast features from grayscale sonar imagery.

    These features are intended as an additional physics-inspired signal
    alongside autoencoder reconstruction error.
    """

    def extract(self, image):
        if image is None:
            raise ValueError("Input image is empty.")

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        gray = cv2.resize(gray, (224, 224))

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        mean_intensity = float(np.mean(blurred))
        std_intensity = float(np.std(blurred))

        dark_threshold = mean_intensity * 0.6
        dark_pixels = blurred < dark_threshold

        shadow_ratio = float(np.mean(dark_pixels))

        return {
            "mean_intensity": mean_intensity,
            "std_intensity": std_intensity,
            "shadow_ratio": shadow_ratio
        }

    def score(self, image):
        features = self.extract(image)

        shadow_ratio = features["shadow_ratio"]
        contrast = features["std_intensity"]

        score = (
            shadow_ratio * 0.5
            + min(contrast / 128.0, 1.0) * 0.5
        )

        return float(score)
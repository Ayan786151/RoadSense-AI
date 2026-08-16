"""
================================================================================
ROAD SENSE AI - ADAPTIVE NIGHT & LOW-LIGHT VISION ENHANCEMENT
MODULE: LOW-LIGHT CONTRAST ENHANCEMENT & GLARE MITIGATION
================================================================================

This module provides adaptive image preprocessing for CCTV footage captured
under low-light, nighttime, shadow, and heavy glare conditions.

MATHEMATICAL & COMPUTER VISION PRINCIPLES:
1. LIGHTING ASSESSMENT:
   - Evaluates mean luminance (L-channel in CIELAB) and RMS contrast.
   - Categorizes lighting: "DAYLIGHT", "LOW_LIGHT", "NIGHT", "HIGH_GLARE".

2. ADAPTIVE LAB-SPACE CLAHE:
   - Converts BGR -> CIELAB color space, isolating luminance L* from chromaticity (a*, b*).
   - Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to L* channel:
     * clipLimit = 2.5 (prevents noise amplification in dark regions).
     * tileGridSize = (8, 8) (local contextual contrast enhancement).
   - Reconstructs BGR with preserved color balance and boosted vehicle silhouette edge gradients.

3. NON-LINEAR GAMMA CORRECTION:
   - For severe nighttime scenes (L < 40): applies power-law lookup table transformation:
     I_out = 255 * (I_in / 255)^gamma, with gamma = 0.70.

4. ZERO DAYTIME DISTORTION:
   - Daytime frames (mean luminance >= 65) bypass enhancement with zero alteration.
================================================================================
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Any, Optional


# Thresholds for lighting classification
LOW_LIGHT_THRESHOLD = 65.0    # Mean L* channel below this triggers enhancement
NIGHT_THRESHOLD = 40.0        # Deep night condition triggers CLAHE + Gamma
GLARE_STD_THRESHOLD = 75.0    # High variance indicative of extreme point-source headlamp glare


def analyze_lighting_conditions(frame: np.ndarray) -> Dict[str, Any]:
    """
    Analyzes frame illumination, contrast, and glare distribution.

    Returns:
        Dict with mean_luminance, contrast_std, lighting_condition, is_low_light.
    """
    if frame is None or frame.size == 0:
        return {
            "mean_luminance": 0.0,
            "contrast_std": 0.0,
            "lighting_condition": "INVALID",
            "is_low_light": False
        }

    # Convert to LAB to decouple luminance from chromaticity
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0]

    mean_l = float(np.mean(l_channel))
    std_l = float(np.std(l_channel))

    if mean_l < NIGHT_THRESHOLD:
        condition = "NIGHT"
        is_low_light = True
    elif mean_l < LOW_LIGHT_THRESHOLD:
        condition = "LOW_LIGHT"
        is_low_light = True
    elif std_l > GLARE_STD_THRESHOLD and mean_l < 90:
        condition = "HIGH_GLARE"
        is_low_light = True
    else:
        condition = "DAYLIGHT"
        is_low_light = False

    return {
        "mean_luminance": round(mean_l, 2),
        "contrast_std": round(std_l, 2),
        "lighting_condition": condition,
        "is_low_light": is_low_light
    }


def enhance_low_light_clahe(
    frame: np.ndarray,
    clip_limit: float = 2.5,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies CLAHE on the luminance channel in CIELAB space.
    Enhances dark vehicle silhouettes without shifting colors.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge([l_enhanced, a_channel, b_channel])
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(frame: np.ndarray, gamma: float = 0.70) -> np.ndarray:
    """
    Applies non-linear power-law gamma correction using a 256-entry lookup table.
    Gamma < 1.0 brightens shadows while preserving highlights.
    """
    inv_gamma = 1.0 / max(gamma, 0.1)
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(frame, table)


def adaptive_preprocess_frame(
    frame: np.ndarray,
    force_enhancement: bool = False
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Assesses frame lighting and adaptively enhances low-light / night scenes
    while leaving daylight frames pristine.

    Returns:
        (preprocessed_frame, lighting_audit)
    """
    audit = analyze_lighting_conditions(frame)

    if not audit["is_low_light"] and not force_enhancement:
        # Daytime: return original frame with zero overhead
        audit["enhancement_applied"] = "NONE (Daylight)"
        return frame, audit

    # Apply CLAHE luminance enhancement
    enhanced = enhance_low_light_clahe(frame, clip_limit=2.5, tile_grid_size=(8, 8))

    # For deep nighttime, also apply subtle gamma lift
    if audit["lighting_condition"] == "NIGHT" or force_enhancement:
        enhanced = apply_gamma_correction(enhanced, gamma=0.75)
        audit["enhancement_applied"] = "CLAHE + Gamma (0.75)"
    else:
        audit["enhancement_applied"] = "LAB-CLAHE"

    return enhanced, audit

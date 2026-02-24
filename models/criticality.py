def calculate_criticality(num_defective_patches: float, w_real: float, w_img: int, a_threshold: float, grid_size: int = 16):
    """
    Computes criticality score.
    W_real: Real width of the component (cm)
    W_img: Width of the image (pixels)
    a_threshold: Critical threshold area (cm^2)
    grid_size: Number of patches along the width
    """
    if w_img == 0 or grid_size == 0 or a_threshold == 0:
        return 0, "Unknown"

    # pixel_to_cm = W_real / W_img
    pixel_to_cm = w_real / w_img
    
    # Linear dimension of a patch in pixels
    p_px = w_img / grid_size
    
    # A_patch = (p × pixel_to_cm)^2 (Area of a single patch in cm^2)
    a_patch = (p_px * pixel_to_cm) ** 2
    
    # A_def = N_defective_patches × A_patch
    a_def = num_defective_patches * a_patch
    
    # C = (A_def / A_threshold) × 100
    c_score = (a_def / a_threshold) * 100
    
    # Severity classification
    if c_score < 30:
        severity = "Minor"
    elif 30 <= c_score < 70:
        severity = "Moderate"
    else:
        severity = "Critical"
        
    return c_score, severity

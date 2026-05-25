from text_drift import check_text_drift
from image_drift import check_image_drift


def check_dual_drift(image, question, algorithm, p_threshold):
    img_result = check_image_drift(image, p_threshold, algorithm)
    txt_result = check_text_drift(question, p_threshold, algorithm)
    
    drift_source = 'none'
    if img_result['is_drift'] and txt_result['is_drift']:
        drift_source = 'both'
    elif img_result['is_drift']:
        drift_source = 'image'
    elif txt_result['is_drift']:
        drift_source = 'text'

    overall_drift = img_result['is_drift'] or txt_result['is_drift']

    return {
        'image_drift': img_result,
        'text_drift': txt_result,
        'overall_drift': overall_drift,
        'drift_source': drift_source,
    }


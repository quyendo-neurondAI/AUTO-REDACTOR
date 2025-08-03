import hashlib
import cv2
import Levenshtein

def compare_with_tolerance(s1, s2, tolerance=2):
    distance = Levenshtein.distance(s1, s2)
    return distance <= tolerance, distance

def find_matching_boxes(matched_text, boxes):
    matched_words = matched_text.split()
    box_texts = [b["text"] for b in boxes]
    indices = []

    for i in range(len(box_texts) - len(matched_words) + 1):
        is_within_tolerance, _ = compare_with_tolerance(
            " ".join(box_texts[i:i + len(matched_words)]),
            matched_text,
            tolerance=round(len(matched_text)*0.1)
        )
        if is_within_tolerance:
            indices.extend(range(i, i + len(matched_words)))
            break
    return indices

def hash_cv2_image(image):
    success, encoded_img = cv2.imencode('.jpg', image)
    return hashlib.md5(encoded_img.tobytes()).hexdigest() if success else None

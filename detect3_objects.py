import os
import json
import cv2
import numpy as np

# =========================================================
# PI-SAFE OPENCV SETTINGS
# =========================================================

cv2.setNumThreads(1)

try:
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


# =========================================================
# CONFIG
# =========================================================

SCAN_DIR = "scan_images"
RESULTS_DIR = "results"
DEBUG_DIR = "debug_aruco"

HEADINGS = ["front", "right", "back", "left"]

# ArUco dictionary
ARUCO_DICT_NAME = "DICT_4X4_50"

# Marker meaning
TARGET_IDS = {0}
OBSTACLE_IDS = {1, 2, 3}

# Only use the front row / lower part of image
ROI_TOP_FRAC = 0.45
ROI_BOT_FRAC = 0.95

# Resize large images for Pi safety
MAX_WIDTH = 800

# Output files
OBJECT_TXT_PATH = os.path.join(RESULTS_DIR, "object_3x3.txt")
OBJECT_JSON_PATH = os.path.join(RESULTS_DIR, "object_results.json")


# =========================================================
# ARUCO SETUP
# =========================================================

def setup_aruco_detector():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is not available. Try installing OpenCV with:\n"
            "sudo apt update\n"
            "sudo apt install python3-opencv\n"
        )

    aruco = cv2.aruco

    dictionary_id = aruco.DICT_4X4_50

    try:
        dictionary = aruco.getPredefinedDictionary(dictionary_id)
    except AttributeError:
        dictionary = aruco.Dictionary_get(dictionary_id)

    try:
        params = aruco.DetectorParameters()
    except AttributeError:
        params = aruco.DetectorParameters_create()

    # Safer on Raspberry Pi
    try:
        params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
    except Exception:
        pass

    # New OpenCV API
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
    else:
        detector = None

    return aruco, dictionary, params, detector


# =========================================================
# HELPERS
# =========================================================

def resize_for_pi(img, max_width=MAX_WIDTH):
    h, w = img.shape[:2]

    if w <= max_width:
        return img

    scale = max_width / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def safe_detect_markers(gray_img, aruco, dictionary, params, detector):
    if gray_img is None or gray_img.size == 0:
        return [], None

    if gray_img.dtype != np.uint8:
        gray_img = gray_img.astype(np.uint8)

    gray_img = np.ascontiguousarray(gray_img)

    try:
        if detector is not None:
            corners, ids, rejected = detector.detectMarkers(gray_img)
        else:
            corners, ids, rejected = aruco.detectMarkers(
                gray_img,
                dictionary,
                parameters=params
            )

        return corners, ids

    except Exception as e:
        print(f"[WARN] ArUco detection failed safely: {e}")
        return [], None


def classify_marker_ids(ids):
    if ids is None or len(ids) == 0:
        return "E", []

    found_ids = [int(x) for x in ids.flatten()]

    for marker_id in found_ids:
        if marker_id in TARGET_IDS:
            return "T", found_ids

    for marker_id in found_ids:
        if marker_id in OBSTACLE_IDS:
            return "O", found_ids

    return "E", found_ids


def update_object_grid_from_heading(object_grid, heading, slot_index, code):
    """
    Local object grid:

        row 0: front-left   front   front-right
        row 1: left         A       right
        row 2: back-left    back    back-right

    Each saved image is taken while robot faces:
        front, right, back, left

    Each image only contributes its front row.
    """

    if code == "E":
        return

    if heading == "front":
        row = 0
        col = slot_index

    elif heading == "right":
        row = slot_index
        col = 2

    elif heading == "back":
        row = 2
        col = 2 - slot_index

    elif heading == "left":
        row = 2 - slot_index
        col = 0

    else:
        return

    if object_grid[row][col] == "E":
        object_grid[row][col] = code


def save_object_grid_txt(object_grid, path):
    with open(path, "w") as f:
        for row in object_grid:
            f.write(" ".join(row) + "\n")


# =========================================================
# MAIN DETECTION FUNCTION
# =========================================================

def detect_objects_aruco():
    """
    Main function to call from runner.

    Reads:
        scan_images/front.jpg
        scan_images/right.jpg
        scan_images/back.jpg
        scan_images/left.jpg

    Writes:
        results/object_3x3.txt
        results/object_results.json
        debug_aruco/*_aruco_debug.jpg

    Returns:
        object_grid
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DEBUG_DIR, exist_ok=True)

    print("\nSTEP 3: DETECT OBJECTS WITH ARUCO")
    print("OpenCV version:", cv2.__version__)
    print("Has cv2.aruco:", hasattr(cv2, "aruco"))

    aruco, dictionary, params, detector = setup_aruco_detector()

    object_grid = [
        ["E", "E", "E"],
        ["E", "A", "E"],
        ["E", "E", "E"],
    ]

    detailed_results = {}

    for heading in HEADINGS:
        img_path = os.path.join(SCAN_DIR, f"{heading}.jpg")
        print(f"\nReading {img_path}")

        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Could not read {img_path}. Skipping.")
            detailed_results[heading] = {"error": "image_not_read"}
            continue

        img = resize_for_pi(img)

        h, w = img.shape[:2]

        y1 = int(h * ROI_TOP_FRAC)
        y2 = int(h * ROI_BOT_FRAC)

        if y2 <= y1:
            print(f"[WARN] Bad ROI for {heading}. Skipping.")
            detailed_results[heading] = {"error": "bad_roi"}
            continue

        roi = img[y1:y2, :]

        if roi.size == 0:
            print(f"[WARN] Empty ROI for {heading}. Skipping.")
            detailed_results[heading] = {"error": "empty_roi"}
            continue

        roi_h, roi_w = roi.shape[:2]
        slot_w = roi_w // 3

        debug_img = roi.copy()
        heading_results = {}

        for slot_index, slot_name in enumerate(["left", "center", "right"]):
            x1 = slot_index * slot_w
            x2 = roi_w if slot_index == 2 else (slot_index + 1) * slot_w

            slot = roi[:, x1:x2]

            if slot.size == 0:
                code = "E"
                ids_found = []
            else:
                gray = cv2.cvtColor(slot, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)

                corners, ids = safe_detect_markers(
                    gray,
                    aruco,
                    dictionary,
                    params,
                    detector
                )

                code, ids_found = classify_marker_ids(ids)

            update_object_grid_from_heading(
                object_grid,
                heading,
                slot_index,
                code
            )

            heading_results[slot_name] = {
                "code": code,
                "ids": ids_found
            }

            if code == "T":
                draw_color = (0, 0, 0)
            elif code == "O":
                draw_color = (0, 0, 255)
            else:
                draw_color = (0, 255, 0)

            cv2.rectangle(
                debug_img,
                (x1, 0),
                (x2, roi_h - 1),
                draw_color,
                2
            )

            cv2.putText(
                debug_img,
                f"{slot_name}: {code} {ids_found}",
                (x1 + 5, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                draw_color,
                1,
                cv2.LINE_AA
            )

            print(f"  {heading} {slot_name}: {code}, ids={ids_found}")

        detailed_results[heading] = heading_results

        debug_path = os.path.join(DEBUG_DIR, f"{heading}_aruco_debug.jpg")
        cv2.imwrite(debug_path, debug_img)

    save_object_grid_txt(object_grid, OBJECT_TXT_PATH)

    with open(OBJECT_JSON_PATH, "w") as f:
        json.dump(
            {
                "object_grid": object_grid,
                "details": detailed_results,
                "target_ids": sorted(list(TARGET_IDS)),
                "obstacle_ids": sorted(list(OBSTACLE_IDS)),
                "roi_top_frac": ROI_TOP_FRAC,
                "roi_bot_frac": ROI_BOT_FRAC,
            },
            f,
            indent=2
        )

    print("\nFinal object grid:")
    for row in object_grid:
        print(" ".join(row))

    print(f"\nSaved: {OBJECT_TXT_PATH}")
    print(f"Saved: {OBJECT_JSON_PATH}")
    print(f"Debug images saved in: {DEBUG_DIR}")

    return object_grid


# =========================================================
# ALIAS FOR RUNNER COMPATIBILITY
# =========================================================

def main():
    return detect_objects_aruco()


if __name__ == "__main__":
    main()

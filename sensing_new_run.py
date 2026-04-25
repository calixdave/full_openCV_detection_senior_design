import os
import sys
import json
import subprocess
import cv2
import numpy as np

# =========================================================
# PI SAFE OPENCV SETTINGS
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
DEBUG_ARUCO_DIR = "debug_aruco"

HEADINGS = ["front", "right", "back", "left"]

# External programs
CAPTURE_SCRIPT = "capture_scan.py"
COLOR_SCRIPT = "detect_colors.py"
MAP_SCRIPT = "map_location.py"

# ArUco object IDs
TARGET_IDS = {0}
OBSTACLE_IDS = {1, 2, 3}

# Front-row ROI only
ROI_TOP_FRAC = 0.45
ROI_BOT_FRAC = 0.95

# Pi safety: resize large images before ArUco detection
MAX_WIDTH = 800

os.makedirs(SCAN_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DEBUG_ARUCO_DIR, exist_ok=True)


# =========================================================
# GENERAL HELPERS
# =========================================================

def run_script(script_name, step_name):
    if not os.path.exists(script_name):
        print(f"[WARN] {script_name} not found. Skipping {step_name}.")
        return False

    print(f"\n{step_name}")
    print(f"Running {script_name}...")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=False
        )

        if result.returncode != 0:
            print(f"[WARN] {script_name} ended with return code {result.returncode}")
            return False

        return True

    except Exception as e:
        print(f"[ERROR] Could not run {script_name}: {e}")
        return False


def check_scan_images():
    missing = []

    for heading in HEADINGS:
        path = os.path.join(SCAN_DIR, f"{heading}.jpg")
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        print("\n[WARN] Missing scan images:")
        for path in missing:
            print("  ", path)
        return False

    print("\nAll scan images found.")
    return True


# =========================================================
# STEP 1: CAPTURE / SCAN
# =========================================================

def step1_scan():
    print("\nSTEP 1: SCAN")

    if os.path.exists(CAPTURE_SCRIPT):
        run_script(CAPTURE_SCRIPT, "STEP 1: SCAN")
    else:
        print("No capture script found. Using existing images in scan_images/.")

    check_scan_images()


# =========================================================
# STEP 2: DETECT COLORS
# =========================================================

def step2_detect_colors():
    print("\nSTEP 2: DETECT COLORS")

    if os.path.exists(COLOR_SCRIPT):
        run_script(COLOR_SCRIPT, "STEP 2: DETECT COLORS")
    else:
        print(f"[WARN] {COLOR_SCRIPT} not found. Skipping color detection.")


# =========================================================
# ARUCO HELPERS
# =========================================================

def setup_aruco_detector():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is not available. "
            "Install OpenCV contrib or use: sudo apt install python3-opencv"
        )

    aruco = cv2.aruco

    try:
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    except AttributeError:
        dictionary = aruco.Dictionary_get(aruco.DICT_4X4_50)

    try:
        params = aruco.DetectorParameters()
    except AttributeError:
        params = aruco.DetectorParameters_create()

    # Pi-safe: avoid unstable corner refinement
    try:
        params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
    except Exception:
        pass

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
    else:
        detector = None

    return aruco, dictionary, params, detector


def resize_for_pi(img):
    h, w = img.shape[:2]

    if w <= MAX_WIDTH:
        return img

    scale = MAX_WIDTH / float(w)
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


# =========================================================
# STEP 3: DETECT OBJECTS WITH SAFE ARUCO
# =========================================================

def step3_detect_objects_aruco():
    print("\nSTEP 3: DETECT OBJECTS WITH SAFE ARUCO")
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

            cv2.rectangle(debug_img, (x1, 0), (x2, roi_h - 1), draw_color, 2)

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

        debug_path = os.path.join(DEBUG_ARUCO_DIR, f"{heading}_aruco_debug.jpg")
        cv2.imwrite(debug_path, debug_img)

    object_txt_path = os.path.join(RESULTS_DIR, "object_3x3.txt")

    with open(object_txt_path, "w") as f:
        for row in object_grid:
            f.write(" ".join(row) + "\n")

    object_json_path = os.path.join(RESULTS_DIR, "object_results.json")

    with open(object_json_path, "w") as f:
        json.dump(
            {
                "object_grid": object_grid,
                "details": detailed_results
            },
            f,
            indent=2
        )

    print("\nFinal object grid:")
    for row in object_grid:
        print(" ".join(row))

    print(f"\nSaved object grid: {object_txt_path}")
    print(f"Saved object details: {object_json_path}")
    print(f"Saved debug images in: {DEBUG_ARUCO_DIR}")

    return object_grid


# =========================================================
# STEP 4: MAP LOCATION
# =========================================================

def step4_map_location():
    print("\nSTEP 4: MAP LOCATION")

    if os.path.exists(MAP_SCRIPT):
        run_script(MAP_SCRIPT, "STEP 4: MAP LOCATION")
    else:
        print(f"[WARN] {MAP_SCRIPT} not found. Skipping map step.")


# =========================================================
# MAIN
# =========================================================

def main():
    print("===================================")
    print("PI SENSING RUNNER")
    print("SAFE ARUCO VERSION")
    print("===================================")

    step1_scan()
    step2_detect_colors()
    step3_detect_objects_aruco()
    step4_map_location()

    print("\nDONE.")


if __name__ == "__main__":
    main()

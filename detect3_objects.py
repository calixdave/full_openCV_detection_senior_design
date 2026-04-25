import os
import json
import time
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
DEBUG_COLOR_DIR = "debug_colors"
DEBUG_ARUCO_DIR = "debug_aruco"

HEADINGS = ["front", "right", "back", "left"]

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

USE_CAMERA_CAPTURE = False   # True = capture new pictures, False = use saved images

# Front-row ROI
ROI_TOP_FRAC = 0.45
ROI_BOT_FRAC = 0.95

MAX_WIDTH = 800

# ArUco marker meaning
TARGET_IDS = {0}
OBSTACLE_IDS = {1, 2, 3}

os.makedirs(SCAN_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DEBUG_COLOR_DIR, exist_ok=True)
os.makedirs(DEBUG_ARUCO_DIR, exist_ok=True)


# =========================================================
# STEP 1: SCAN / CAPTURE
# =========================================================

def capture_scan_images():
    print("\nSTEP 1: SCAN")

    if not USE_CAMERA_CAPTURE:
        print("Using saved images in scan_images/.")
        check_scan_images()
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")

    print("Press c to capture each heading.")
    print("Order: front, right, back, left")
    print("Press q to quit.")

    index = 0

    while index < len(HEADINGS):
        heading = HEADINGS[index]

        ret, frame = cap.read()

        if not ret or frame is None:
            print("[WARN] Could not read frame.")
            continue

        display = frame.copy()

        cv2.putText(
            display,
            f"Capture: {heading.upper()}  ({index + 1}/4)",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        draw_roi_guides(display)

        cv2.imshow("Capture Scan", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            save_path = os.path.join(SCAN_DIR, f"{heading}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"Saved {save_path}")
            index += 1
            time.sleep(0.5)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    check_scan_images()


def draw_roi_guides(img):
    h, w = img.shape[:2]
    y1 = int(h * ROI_TOP_FRAC)
    y2 = int(h * ROI_BOT_FRAC)

    cv2.rectangle(img, (0, y1), (w - 1, y2), (0, 255, 0), 2)

    slot_w = w // 3

    for i in range(1, 3):
        x = i * slot_w
        cv2.line(img, (x, y1), (x, y2), (0, 255, 0), 2)


def check_scan_images():
    missing = []

    for heading in HEADINGS:
        path = os.path.join(SCAN_DIR, f"{heading}.jpg")
        if not os.path.exists(path):
            missing.append(path)

    if missing:
        print("[ERROR] Missing scan images:")
        for path in missing:
            print(" ", path)
        raise FileNotFoundError("Missing one or more scan images.")

    print("[OK] All scan images found.")


# =========================================================
# IMAGE HELPERS
# =========================================================

def resize_for_pi(img, max_width=MAX_WIDTH):
    h, w = img.shape[:2]

    if w <= max_width:
        return img

    scale = max_width / float(w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def get_front_roi(img):
    img = resize_for_pi(img)

    h, w = img.shape[:2]
    y1 = int(h * ROI_TOP_FRAC)
    y2 = int(h * ROI_BOT_FRAC)

    roi = img[y1:y2, :]

    return roi


def split_roi_into_slots(roi):
    h, w = roi.shape[:2]
    slot_w = w // 3

    slots = []

    for i in range(3):
        x1 = i * slot_w
        x2 = w if i == 2 else (i + 1) * slot_w
        slots.append((i, x1, x2, roi[:, x1:x2]))

    return slots


# =========================================================
# STEP 2: DETECT COLORS
# =========================================================

def classify_tile_color(slot_img):
    """
    Simple OpenCV HSV color classifier.
    Letters:
        B = blue
        G = green
        R = red
        Y = yellow
        M = pink/magenta
        P = purple
        ? = unknown
    """

    if slot_img is None or slot_img.size == 0:
        return "?"

    h, w = slot_img.shape[:2]

    # Use center crop to avoid tile borders/noise
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)
    y1 = int(h * 0.25)
    y2 = int(h * 0.75)

    crop = slot_img[y1:y2, x1:x2]

    if crop.size == 0:
        return "?"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    mask = (S > 40) & (V > 60)

    if np.count_nonzero(mask) < 50:
        return "?"

    mean_h = float(np.mean(H[mask]))

    if 95 <= mean_h <= 130:
        return "B"
    elif 35 <= mean_h <= 85:
        return "G"
    elif 20 <= mean_h < 35:
        return "Y"
    elif mean_h <= 10 or mean_h >= 170:
        return "R"
    elif 135 <= mean_h < 160:
        return "M"
    elif 130 <= mean_h < 150:
        return "P"
    else:
        return "?"


def update_color_grid_from_heading(color_grid, heading, slot_index, code):
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

    if color_grid[row][col] == "?":
        color_grid[row][col] = code


def detect_colors():
    print("\nSTEP 2: DETECT COLORS")

    color_grid = [
        ["?", "?", "?"],
        ["?", "A", "?"],
        ["?", "?", "?"],
    ]

    details = {}

    for heading in HEADINGS:
        img_path = os.path.join(SCAN_DIR, f"{heading}.jpg")
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Could not read {img_path}")
            details[heading] = {"error": "image_not_read"}
            continue

        roi = get_front_roi(img)
        debug_img = roi.copy()

        heading_details = {}

        for slot_index, x1, x2, slot in split_roi_into_slots(roi):
            code = classify_tile_color(slot)
            update_color_grid_from_heading(color_grid, heading, slot_index, code)

            slot_name = ["left", "center", "right"][slot_index]
            heading_details[slot_name] = code

            cv2.rectangle(debug_img, (x1, 0), (x2, roi.shape[0] - 1), (0, 255, 0), 2)
            cv2.putText(
                debug_img,
                f"{slot_name}:{code}",
                (x1 + 5, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

            print(f"  {heading} {slot_name}: {code}")

        details[heading] = heading_details

        debug_path = os.path.join(DEBUG_COLOR_DIR, f"{heading}_color_debug.jpg")
        cv2.imwrite(debug_path, debug_img)

    save_grid_txt(color_grid, os.path.join(RESULTS_DIR, "color_3x3.txt"))

    with open(os.path.join(RESULTS_DIR, "color_results.json"), "w") as f:
        json.dump(
            {
                "color_grid": color_grid,
                "details": details
            },
            f,
            indent=2
        )

    print("\nColor grid:")
    print_grid(color_grid)

    return color_grid


# =========================================================
# STEP 3: DETECT OBJECTS WITH SAFE ARUCO
# =========================================================

def setup_aruco_detector():
    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is not available. Install with:\n"
            "sudo apt update\n"
            "sudo apt install python3-opencv"
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

    try:
        params.cornerRefinementMethod = aruco.CORNER_REFINE_NONE
    except Exception:
        pass

    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, params)
    else:
        detector = None

    return aruco, dictionary, params, detector


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


def detect_objects_aruco():
    print("\nSTEP 3: DETECT OBJECTS WITH SAFE ARUCO")
    print("OpenCV version:", cv2.__version__)
    print("Has cv2.aruco:", hasattr(cv2, "aruco"))

    aruco, dictionary, params, detector = setup_aruco_detector()

    object_grid = [
        ["E", "E", "E"],
        ["E", "A", "E"],
        ["E", "E", "E"],
    ]

    details = {}

    for heading in HEADINGS:
        img_path = os.path.join(SCAN_DIR, f"{heading}.jpg")
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARN] Could not read {img_path}")
            details[heading] = {"error": "image_not_read"}
            continue

        roi = get_front_roi(img)
        debug_img = roi.copy()

        heading_details = {}

        for slot_index, x1, x2, slot in split_roi_into_slots(roi):
            if slot is None or slot.size == 0:
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

            update_object_grid_from_heading(object_grid, heading, slot_index, code)

            slot_name = ["left", "center", "right"][slot_index]
            heading_details[slot_name] = {
                "code": code,
                "ids": ids_found
            }

            if code == "T":
                draw_color = (0, 0, 0)
            elif code == "O":
                draw_color = (0, 0, 255)
            else:
                draw_color = (0, 255, 0)

            cv2.rectangle(debug_img, (x1, 0), (x2, roi.shape[0] - 1), draw_color, 2)
            cv2.putText(
                debug_img,
                f"{slot_name}:{code} {ids_found}",
                (x1 + 5, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                draw_color,
                1,
                cv2.LINE_AA
            )

            print(f"  {heading} {slot_name}: {code}, ids={ids_found}")

        details[heading] = heading_details

        debug_path = os.path.join(DEBUG_ARUCO_DIR, f"{heading}_aruco_debug.jpg")
        cv2.imwrite(debug_path, debug_img)

    save_grid_txt(object_grid, os.path.join(RESULTS_DIR, "object_3x3.txt"))

    with open(os.path.join(RESULTS_DIR, "object_results.json"), "w") as f:
        json.dump(
            {
                "object_grid": object_grid,
                "details": details,
                "target_ids": sorted(list(TARGET_IDS)),
                "obstacle_ids": sorted(list(OBSTACLE_IDS))
            },
            f,
            indent=2
        )

    print("\nObject grid:")
    print_grid(object_grid)

    return object_grid


# =========================================================
# STEP 4: MAP LOCATION
# =========================================================

BIG_GRID = [
    ["B", "G", "R", "Y", "M", "P"],
    ["G", "Y", "B", "M", "R", "P"],
    ["R", "B", "M", "G", "P", "Y"],
    ["Y", "M", "G", "P", "B", "R"],
    ["M", "R", "P", "B", "Y", "G"],
    ["P", "P", "Y", "R", "G", "M"],
]


def rotate_grid_cw(grid):
    return [list(row) for row in zip(*grid[::-1])]


def rotate_grid_n_times(grid, n):
    out = [row[:] for row in grid]
    for _ in range(n):
        out = rotate_grid_cw(out)
    return out


def count_match(local_grid, window):
    score = 0
    total = 0

    for r in range(3):
        for c in range(3):
            val = local_grid[r][c]

            if val in ["?", "A"]:
                continue

            total += 1

            if val == window[r][c]:
                score += 1

    return score, total


def get_window(big_grid, top, left):
    return [
        big_grid[top][left:left + 3],
        big_grid[top + 1][left:left + 3],
        big_grid[top + 2][left:left + 3],
    ]


def map_location(color_grid, object_grid):
    print("\nSTEP 4: MAP LOCATION")

    best = None
    matches = []

    for rot in range(4):
        test_grid = rotate_grid_n_times(color_grid, rot)

        for top in range(len(BIG_GRID) - 2):
            for left in range(len(BIG_GRID[0]) - 2):
                window = get_window(BIG_GRID, top, left)
                score, total = count_match(test_grid, window)

                item = {
                    "rotation_cw": rot,
                    "top": top,
                    "left": left,
                    "score": score,
                    "total": total,
                    "window": window
                }

                matches.append(item)

                if best is None or score > best["score"]:
                    best = item

    if best is None:
        result = {
            "status": "no_match",
            "color_grid": color_grid,
            "object_grid": object_grid
        }
    else:
        result = {
            "status": "matched",
            "best_match": best,
            "color_grid": color_grid,
            "object_grid": object_grid
        }

    with open(os.path.join(RESULTS_DIR, "map_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    print("\nBest map result:")
    print(json.dumps(result, indent=2))

    return result


# =========================================================
# STEP 5: FINAL OUTPUT
# =========================================================

def generate_final_output(color_grid, object_grid, map_result):
    print("\nSTEP 5: GENERATE FINAL OUTPUT")

    final_path = os.path.join(RESULTS_DIR, "final_result.txt")

    direction = "U"

    if map_result.get("status") == "matched":
        rot = map_result["best_match"]["rotation_cw"]

        # Approximate direction based on rotation
        direction_lookup = {
            0: "U",
            1: "R",
            2: "D",
            3: "L",
        }

        direction = direction_lookup.get(rot, "U")

    compact = flatten_grid_without_agent(color_grid) + direction

    with open(final_path, "w") as f:
        f.write("COLOR GRID:\n")
        for row in color_grid:
            f.write(" ".join(row) + "\n")

        f.write("\nOBJECT GRID:\n")
        for row in object_grid:
            f.write(" ".join(row) + "\n")

        f.write("\nDIRECTION:\n")
        f.write(direction + "\n")

        f.write("\nCOMPACT:\n")
        f.write(compact + "\n")

    print(f"Saved final result: {final_path}")
    print("Compact output:", compact)

    return compact


def flatten_grid_without_agent(grid):
    out = []

    for r in range(3):
        for c in range(3):
            if r == 1 and c == 1:
                continue
            out.append(grid[r][c])

    return "".join(out)


# =========================================================
# GENERAL SAVE/PRINT HELPERS
# =========================================================

def save_grid_txt(grid, path):
    with open(path, "w") as f:
        for row in grid:
            f.write(" ".join(row) + "\n")


def print_grid(grid):
    for row in grid:
        print(" ".join(row))


# =========================================================
# MAIN RUNNER
# =========================================================

def main():
    print("===================================")
    print("PI SENSING FULL SINGLE-FILE RUNNER")
    print("===================================")

    capture_scan_images()

    color_grid = detect_colors()

    object_grid = detect_objects_aruco()

    map_result = map_location(color_grid, object_grid)

    compact = generate_final_output(color_grid, object_grid, map_result)

    print("\n===================================")
    print("DONE")
    print("===================================")

    return compact


if __name__ == "__main__":
    main()

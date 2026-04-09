import cv2
import os
import time
import json
import numpy as np

# =========================================================
# GLOBAL PATHS
# =========================================================

SCAN_DIR = "scan_images"
DEBUG_COLOR_DIR = "debug_tiles"
DEBUG_OBJECT_DIR = "debug_objects"
RESULTS_DIR = "results"

FINAL_COMPACT_FILE = os.path.join(RESULTS_DIR, "compact_17char.txt")

HEADINGS = ["front", "right", "back", "left"]

# =========================================================
# CAPTURE CONFIG
# =========================================================

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

CAPTURE_ROI_TOP_FRAC = 0.34
CAPTURE_ROI_BOT_FRAC = 0.94
CAPTURE_SLOT_PAD_X_FRAC = 0.03
CAPTURE_SLOT_PAD_Y_FRAC = 0.06

# =========================================================
# COLOR DETECTION CONFIG
# =========================================================

COLOR_ROI_TOP_FRAC = 0.55
COLOR_ROI_BOT_FRAC = 0.95
COLOR_SLOT_PAD_X_FRAC = 0.03
COLOR_SLOT_PAD_Y_FRAC = 0.06

# =========================================================
# OBJECT DETECTION CONFIG
# =========================================================

OBJECT_ROI_TOP_FRAC = 0.34
OBJECT_ROI_BOT_FRAC = 0.94
OBJECT_SLOT_PAD_X_FRAC = 0.03
OBJECT_SLOT_PAD_Y_FRAC = 0.06

WHITE_MIN = 126
BLACK_MAX = 95
RED_S_MIN = 95
RED_V_MIN = 120

WHITE_RATIO_TH = 0.35
RED_RATIO_TH = 0.18
BLACK_RATIO_TH = 0.14

EMPTY_WHITE_TH = 0.80
EMPTY_RED_TH = 0.80
EMPTY_BLACK_TH = 0.90

CANNY1 = 40
CANNY2 = 120
HOUGH_TH = 30
MIN_LINE = 8
MAX_GAP = 40
BLUR_ODD = 1

# =========================================================
# MAP CONFIG
# =========================================================

BIG_GRID = [
    ['G', 'R', 'P', 'Y', 'P', 'P'],
    ['P', 'Y', 'B', 'R', 'M', 'G'],
    ['P', 'P', 'Y', 'R', 'R', 'B'],
    ['M', 'G', 'G', 'M', 'Y', 'Y'],
    ['B', 'M', 'Y', 'M', 'M', 'B'],
    ['R', 'G', 'G', 'B', 'R', 'B'],
]

MIN_KNOWN_NEIGHBORS = 5
MAX_MISMATCHES = 4

SCAN_START_LOCAL = "FRONT"
SCAN_SWEEP = "cw"
NUM_VIEWS = 4

HEADING_TO_POSITIONS = {
    "front": [(-1, +1), (0, +1), (+1, +1)],
    "right": [(+1, +1), (+1, 0), (+1, -1)],
    "back":  [(+1, -1), (0, -1), (-1, -1)],
    "left":  [(-1, -1), (-1, 0), (-1, +1)],
}


# =========================================================
# COMMON HELPERS
# =========================================================

def ensure_dirs():
    os.makedirs(SCAN_DIR, exist_ok=True)
    os.makedirs(DEBUG_COLOR_DIR, exist_ok=True)
    os.makedirs(DEBUG_OBJECT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def put_text(img, text, y, scale=0.7, thickness=2):
    cv2.putText(
        img,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA
    )


def matrix_rows_from_grid(final_grid):
    rows = []
    for row in [1, 0, -1]:
        vals = []
        for col in [-1, 0, 1]:
            vals.append(final_grid.get((col, row), "?"))
        rows.append(vals)
    return rows


def pretty_matrix(mat):
    return "\n".join(" ".join(row) for row in mat)


def save_matrix_txt(path, final_grid):
    rows = matrix_rows_from_grid(final_grid)
    with open(path, "w") as f:
        for row in rows:
            f.write(" ".join(row) + "\n")


def read_local_3x3(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find file: {path}")

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) != 3:
                raise ValueError(f"Each row must have 3 entries. Bad row: {line}")
            rows.append([p.upper() for p in parts])

    if len(rows) != 3:
        raise ValueError(f"Expected 3 rows in local 3x3 file, found {len(rows)}")

    return rows


# =========================================================
# 1) CAPTURE SCAN
# =========================================================

def draw_capture_slot_guides(img):
    h, w = img.shape[:2]
    y0 = int(CAPTURE_ROI_TOP_FRAC * h)
    y1 = int(CAPTURE_ROI_BOT_FRAC * h)

    if y1 <= y0:
        return img

    out = img.copy()
    cv2.rectangle(out, (0, y0), (w - 1, y1), (255, 255, 0), 1)

    band_h = y1 - y0
    for i in range(3):
        sx0 = int(i * w / 3)
        sx1 = int((i + 1) * w / 3)

        pad_x = int(CAPTURE_SLOT_PAD_X_FRAC * (sx1 - sx0))
        pad_y = int(CAPTURE_SLOT_PAD_Y_FRAC * band_h)

        cx0 = max(0, sx0 + pad_x)
        cx1 = min(w, sx1 - pad_x)
        cy0 = max(0, y0 + pad_y)
        cy1 = min(h, y1 - pad_y)

        cv2.rectangle(out, (cx0, cy0), (cx1, cy1), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"slot {i}",
            (cx0 + 5, cy0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return out


def capture_scan():
    print("\n=== STEP 1: CAPTURE SCAN ===")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("ERROR: Could not open camera.")

    print("Camera opened.")
    print("Press 'c' to capture each heading.")
    print("Capture order: front -> right -> back -> left")
    print("Program will continue automatically after all 4 captures.")

    idx = 0
    last_capture_msg = ""

    while idx < len(HEADINGS):
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError("ERROR: Failed to read frame from camera.")

        display = draw_capture_slot_guides(frame)

        current_heading = HEADINGS[idx]
        put_text(display, f"Current heading: {current_heading}", 30, 0.9, 2)
        put_text(display, "Press 'c' to capture this view", 65)
        put_text(display, "Rotate camera/robot manually before each capture", 95)
        put_text(display, "Green boxes = approximate detection slots", 125)

        if last_capture_msg:
            put_text(display, last_capture_msg, FRAME_HEIGHT - 20, 0.65, 2)

        cv2.imshow("capture_scan", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):
            heading = HEADINGS[idx]
            filename = os.path.join(SCAN_DIR, f"{heading}.jpg")

            ok = cv2.imwrite(filename, frame)
            if not ok:
                cap.release()
                cv2.destroyAllWindows()
                raise RuntimeError(f"ERROR: Failed to save {filename}")

            print(f"Saved: {filename}")
            last_capture_msg = f"Saved {heading}.jpg"
            idx += 1
            time.sleep(0.4)

    cap.release()
    cv2.destroyAllWindows()
    print("All 4 captures completed.")
    print("Moving to next step...")


# =========================================================
# 2) DETECT COLORS
# =========================================================

def get_three_slot_rois_color(img):
    h, w = img.shape[:2]

    y0 = int(COLOR_ROI_TOP_FRAC * h)
    y1 = int(COLOR_ROI_BOT_FRAC * h)

    if y1 <= y0:
        return []

    band = img[y0:y1, :]
    bh, bw = band.shape[:2]
    slots = []

    for i in range(3):
        sx0 = int(i * bw / 3)
        sx1 = int((i + 1) * bw / 3)

        pad_x = int(COLOR_SLOT_PAD_X_FRAC * (sx1 - sx0))
        pad_y = int(COLOR_SLOT_PAD_Y_FRAC * bh)

        cx0 = max(0, sx0 + pad_x)
        cx1 = min(bw, sx1 - pad_x)
        cy0 = max(0, pad_y)
        cy1 = min(bh, bh - pad_y)

        crop = band[cy0:cy1, cx0:cx1]
        slots.append(crop)

    return slots


def center_crop(img, frac=0.50):
    h, w = img.shape[:2]
    y0 = int((1.0 - frac) * 0.5 * h)
    y1 = int(h - y0)
    x0 = int((1.0 - frac) * 0.5 * w)
    x1 = int(w - x0)
    return img[y0:y1, x0:x1]


def classify_color_opencv(tile_bgr):
    if tile_bgr is None or tile_bgr.size == 0:
        return "unknown", "?", {"reason": "empty_roi"}

    roi = center_crop(tile_bgr, 0.50)
    if roi.size == 0:
        return "unknown", "?", {"reason": "bad_center_crop"}

    roi = cv2.GaussianBlur(roi, (5, 5), 0)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    valid = (S >= 45) & (V >= 45)
    valid_ratio = float(np.count_nonzero(valid)) / float(valid.size)

    if valid_ratio < 0.12:
        return "unknown", "?", {
            "reason": "too_little_colored_area",
            "valid_ratio": round(valid_ratio, 4)
        }

    hvals = H[valid]
    s_mean = float(np.mean(S[valid]))
    v_mean = float(np.mean(V[valid]))
    h_mean = float(np.mean(hvals))
    a_mean = float(np.mean(A[valid]))
    b_mean = float(np.mean(B[valid]))

    red_mask = (((H <= 10) | (H >= 170)) & valid)
    yellow_mask = ((H >= 18) & (H <= 38) & valid)
    green_mask = ((H >= 40) & (H <= 90) & valid)
    blue_mask = ((H >= 95) & (H <= 135) & valid)
    pm_mask = ((H >= 136) & (H <= 169) & valid)

    ratios = {
        "red": float(np.count_nonzero(red_mask)) / float(valid.size),
        "yellow": float(np.count_nonzero(yellow_mask)) / float(valid.size),
        "green": float(np.count_nonzero(green_mask)) / float(valid.size),
        "blue": float(np.count_nonzero(blue_mask)) / float(valid.size),
        "pm": float(np.count_nonzero(pm_mask)) / float(valid.size),
    }

    label = "unknown"
    ch = "?"

    best_basic = max(ratios, key=ratios.get)
    best_ratio = ratios[best_basic]

    if best_ratio >= 0.18:
        if best_basic == "red":
            label, ch = "red", "R"
        elif best_basic == "yellow":
            label, ch = "yellow", "Y"
        elif best_basic == "green":
            label, ch = "green", "G"
        elif best_basic == "blue":
            label, ch = "blue", "B"
        elif best_basic == "pm":
            if v_mean >= 125 or b_mean >= 145:
                label, ch = "pink", "M"
            else:
                label, ch = "purple", "P"

    if label == "unknown":
        if a_mean >= 145 and b_mean >= 135 and v_mean >= 110:
            label, ch = "pink", "M"
        elif a_mean >= 145 and b_mean < 135:
            label, ch = "purple", "P"

    metrics = {
        "valid_ratio": round(valid_ratio, 4),
        "h_mean": round(h_mean, 2),
        "s_mean": round(s_mean, 2),
        "v_mean": round(v_mean, 2),
        "lab_a_mean": round(a_mean, 2),
        "lab_b_mean": round(b_mean, 2),
        "ratios": {k: round(v, 4) for k, v in ratios.items()}
    }

    return label, ch, metrics


def detect_colors():
    print("\n=== STEP 2: DETECT COLORS ===")

    final_grid = {
        (-1, +1): "?",
        ( 0, +1): "?",
        (+1, +1): "?",
        (-1,  0): "?",
        ( 0,  0): "A",
        (+1,  0): "?",
        (-1, -1): "?",
        ( 0, -1): "?",
        (+1, -1): "?",
    }

    detailed = {}

    for heading in HEADINGS:
        path = os.path.join(SCAN_DIR, f"{heading}.jpg")

        if not os.path.exists(path):
            raise FileNotFoundError(f"ERROR: Missing image: {path}")

        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"ERROR: Could not read image: {path}")

        slots = get_three_slot_rois_color(img)
        if len(slots) != 3:
            raise RuntimeError(f"ERROR: Could not build 3 slots for heading: {heading}")

        heading_info = []
        print(f"\nHeading: {heading}")

        for i, tile in enumerate(slots):
            dbg_name = os.path.join(DEBUG_COLOR_DIR, f"{heading}_slot{i}.jpg")
            cv2.imwrite(dbg_name, tile)

            label, ch, metrics = classify_color_opencv(tile)
            pos = HEADING_TO_POSITIONS[heading][i]
            final_grid[pos] = ch

            print(f"  slot {i}: label={label}, char={ch}, saved={dbg_name}")
            print(f"    metrics: {metrics}")

            heading_info.append({
                "slot_index": i,
                "pos": [pos[0], pos[1]],
                "label": label,
                "char": ch,
                "debug_crop": dbg_name,
                "metrics": metrics
            })

        detailed[heading] = heading_info

    out = {
        "center": [0, 0],
        "agent": "A",
        "grid_letters": {
            f"{c},{r}": final_grid[(c, r)]
            for (c, r) in final_grid
        },
        "per_heading": detailed
    }

    json_path = os.path.join(RESULTS_DIR, "color_results.json")
    txt_path = os.path.join(RESULTS_DIR, "local_color_3x3.txt")

    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    save_matrix_txt(txt_path, final_grid)

    print("\nFinal 3x3 color matrix:")
    print(pretty_matrix(matrix_rows_from_grid(final_grid)))
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


# =========================================================
# 3) DETECT OBJECTS
# =========================================================

def get_three_slot_rois_object(img):
    h, w = img.shape[:2]

    y0 = int(OBJECT_ROI_TOP_FRAC * h)
    y1 = int(OBJECT_ROI_BOT_FRAC * h)

    if y1 <= y0:
        return []

    band = img[y0:y1, :]
    bh, bw = band.shape[:2]
    slots = []

    for i in range(3):
        sx0 = int(i * bw / 3)
        sx1 = int((i + 1) * bw / 3)

        pad_x = int(OBJECT_SLOT_PAD_X_FRAC * (sx1 - sx0))
        pad_y = int(OBJECT_SLOT_PAD_Y_FRAC * bh)

        cx0 = max(0, sx0 + pad_x)
        cx1 = min(bw, sx1 - pad_x)
        cy0 = max(0, pad_y)
        cy1 = min(bh, bh - pad_y)

        crop = band[cy0:cy1, cx0:cx1]
        slots.append(crop)

    return slots


def detect_one_object_slot(slot_bgr):
    if slot_bgr is None or slot_bgr.size == 0:
        return "?", {}

    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2GRAY)

    if BLUR_ODD > 1:
        k = BLUR_ODD if BLUR_ODD % 2 == 1 else BLUR_ODD + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    lower_red1 = np.array([0, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red1, red2)
    red_ratio = float(np.count_nonzero(red_mask)) / red_mask.size

    white_mask = cv2.inRange(gray, WHITE_MIN, 255)
    white_ratio = float(np.count_nonzero(white_mask)) / white_mask.size

    black_mask = cv2.inRange(gray, 0, BLACK_MAX)
    black_ratio = float(np.count_nonzero(black_mask)) / black_mask.size

    edges = cv2.Canny(gray, CANNY1, CANNY2)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(1, HOUGH_TH),
        minLineLength=max(1, MIN_LINE),
        maxLineGap=max(0, MAX_GAP)
    )

    diag_pos = 0
    diag_neg = 0

    if lines is not None:
        for ln in lines[:, 0]:
            x1, y1, x2, y2 = ln
            dx = x2 - x1
            dy = y2 - y1

            if dx == 0:
                continue

            ang = np.degrees(np.arctan2(dy, dx))

            if 25 <= ang <= 65:
                diag_pos += 1
            elif -65 <= ang <= -25:
                diag_neg += 1

    has_x_shape = (diag_pos >= 1 and diag_neg >= 1)

    metrics = {
        "red_ratio": round(red_ratio, 4),
        "white_ratio": round(white_ratio, 4),
        "black_ratio": round(black_ratio, 4),
        "diag_pos": int(diag_pos),
        "diag_neg": int(diag_neg),
        "has_x_shape": bool(has_x_shape),
    }

    if white_ratio > WHITE_RATIO_TH and red_ratio > RED_RATIO_TH and has_x_shape:
        return "O", metrics

    if white_ratio > WHITE_RATIO_TH and black_ratio > BLACK_RATIO_TH and has_x_shape:
        return "T", metrics

    if white_ratio < EMPTY_WHITE_TH and red_ratio < EMPTY_RED_TH and black_ratio < EMPTY_BLACK_TH:
        return "E", metrics

    return "?", metrics


def detect_objects():
    print("\n=== STEP 3: DETECT OBJECTS ===")

    final_grid = {
        (-1, +1): "?",
        ( 0, +1): "?",
        (+1, +1): "?",
        (-1,  0): "?",
        ( 0,  0): "A",
        (+1,  0): "?",
        (-1, -1): "?",
        ( 0, -1): "?",
        (+1, -1): "?",
    }

    detailed = {}

    for heading in HEADINGS:
        path = os.path.join(SCAN_DIR, f"{heading}.jpg")

        if not os.path.exists(path):
            raise FileNotFoundError(f"ERROR: Missing image: {path}")

        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"ERROR: Could not read image: {path}")

        slots = get_three_slot_rois_object(img)
        if len(slots) != 3:
            raise RuntimeError(f"ERROR: Could not build 3 slots for heading: {heading}")

        heading_info = []
        print(f"\nHeading: {heading}")

        for i, tile in enumerate(slots):
            dbg_name = os.path.join(DEBUG_OBJECT_DIR, f"{heading}_slot{i}.jpg")
            cv2.imwrite(dbg_name, tile)

            obj_char, metrics = detect_one_object_slot(tile)
            pos = HEADING_TO_POSITIONS[heading][i]
            final_grid[pos] = obj_char

            print(f"  slot {i}: object={obj_char}, saved={dbg_name}")
            print(f"    metrics: {metrics}")

            heading_info.append({
                "slot_index": i,
                "pos": [pos[0], pos[1]],
                "object": obj_char,
                "debug_crop": dbg_name,
                "metrics": metrics
            })

        detailed[heading] = heading_info

    out = {
        "center": [0, 0],
        "agent": "A",
        "grid_objects": {
            f"{c},{r}": final_grid[(c, r)]
            for (c, r) in final_grid
        },
        "per_heading": detailed
    }

    json_path = os.path.join(RESULTS_DIR, "object_results.json")
    txt_path = os.path.join(RESULTS_DIR, "local_object_3x3.txt")

    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)

    save_matrix_txt(txt_path, final_grid)

    print("\nFinal 3x3 object matrix:")
    print(pretty_matrix(matrix_rows_from_grid(final_grid)))
    print(f"Saved: {json_path}")
    print(f"Saved: {txt_path}")


# =========================================================
# 4) MAP LOCATION
# =========================================================

def rotate_3x3_ccw(mat):
    return [
        [mat[0][2], mat[1][2], mat[2][2]],
        [mat[0][1], mat[1][1], mat[2][1]],
        [mat[0][0], mat[1][0], mat[2][0]],
    ]


def rotate_n_ccw(mat, n):
    out = [row[:] for row in mat]
    for _ in range(n % 4):
        out = rotate_3x3_ccw(out)
    return out


def get_window_3x3(grid, center_r, center_c):
    rows = len(grid)
    cols = len(grid[0])

    if center_r - 1 < 0 or center_r + 1 >= rows:
        return None
    if center_c - 1 < 0 or center_c + 1 >= cols:
        return None

    return [
        [grid[center_r - 1][center_c - 1], grid[center_r - 1][center_c], grid[center_r - 1][center_c + 1]],
        [grid[center_r][center_c - 1],     'A',                          grid[center_r][center_c + 1]],
        [grid[center_r + 1][center_c - 1], grid[center_r + 1][center_c], grid[center_r + 1][center_c + 1]],
    ]


def score_match(local_3x3, window_3x3):
    known = 0
    matches = 0
    mismatches = 0

    for r in range(3):
        for c in range(3):
            lv = local_3x3[r][c]
            wv = window_3x3[r][c]

            if lv == 'A':
                continue
            if lv == '?':
                continue

            known += 1
            if lv == wv:
                matches += 1
            else:
                mismatches += 1

    return {
        "known": known,
        "matches": matches,
        "mismatches": mismatches,
        "score": matches
    }


def rotation_to_facing(rotation_ccw_deg):
    mapping = {
        0: "UP",
        90: "RIGHT",
        180: "DOWN",
        270: "LEFT",
    }
    return mapping[rotation_ccw_deg]


def rotate_direction(direction, steps_ccw):
    dirs = ["UP", "LEFT", "DOWN", "RIGHT"]
    idx = dirs.index(direction)
    return dirs[(idx + steps_ccw) % 4]


def get_scan_order(scan_start_local="FRONT", scan_sweep="cw", num_views=4):
    scan_start_local = scan_start_local.upper()
    scan_sweep = scan_sweep.lower()

    if scan_sweep == "cw":
        base_order = ["FRONT", "RIGHT", "BACK", "LEFT"]
    elif scan_sweep == "ccw":
        base_order = ["FRONT", "LEFT", "BACK", "RIGHT"]
    else:
        raise ValueError(f"scan_sweep must be 'cw' or 'ccw', got: {scan_sweep}")

    if scan_start_local not in base_order:
        raise ValueError(f"scan_start_local must be one of {base_order}, got: {scan_start_local}")

    start_idx = base_order.index(scan_start_local)
    ordered = base_order[start_idx:] + base_order[:start_idx]
    return ordered[:num_views]


def local_heading_to_map_direction(start_map_direction, local_heading):
    local_heading = local_heading.upper()

    local_steps_ccw = {
        "FRONT": 0,
        "LEFT": 1,
        "BACK": 2,
        "RIGHT": 3,
    }

    return rotate_direction(start_map_direction, local_steps_ccw[local_heading])


def get_final_camera_direction_after_scan(start_map_direction, scan_start_local="FRONT", scan_sweep="cw", num_views=4):
    order = get_scan_order(scan_start_local, scan_sweep, num_views)
    final_local_heading = order[-1]
    final_map_direction = local_heading_to_map_direction(start_map_direction, final_local_heading)
    return final_local_heading, final_map_direction


def direction_to_char(direction):
    mapping = {
        "UP": "U",
        "RIGHT": "R",
        "DOWN": "D",
        "LEFT": "L",
    }
    return mapping[direction]


def build_compact_17char(color_3x3, object_3x3, final_direction):
    out = []

    for r in range(3):
        for c in range(3):
            if r == 1 and c == 1:
                continue

            color_char = str(color_3x3[r][c]).strip().upper()[:1]
            obj_char = str(object_3x3[r][c]).strip().upper()[:1]
            out.append(color_char + obj_char)

    out.append(direction_to_char(final_direction))
    return "".join(out)


def find_best_match(local_3x3, big_grid):
    rows = len(big_grid)
    cols = len(big_grid[0])
    candidates = []

    for rot_steps in range(4):
        rotated = rotate_n_ccw(local_3x3, rot_steps)
        rotation_ccw_deg = rot_steps * 90

        for center_r in range(1, rows - 1):
            for center_c in range(1, cols - 1):
                window = get_window_3x3(big_grid, center_r, center_c)
                if window is None:
                    continue

                s = score_match(rotated, window)

                if s["known"] < MIN_KNOWN_NEIGHBORS:
                    continue
                if s["mismatches"] > MAX_MISMATCHES:
                    continue

                candidates.append({
                    "center_row": center_r,
                    "center_col": center_c,
                    "rotation_ccw_deg": rotation_ccw_deg,
                    "facing": rotation_to_facing(rotation_ccw_deg),
                    "known": s["known"],
                    "matches": s["matches"],
                    "mismatches": s["mismatches"],
                    "score": s["score"],
                    "rotated_local": rotated,
                    "window": window,
                })

    if not candidates:
        return None, []

    candidates.sort(
        key=lambda x: (x["score"], -x["mismatches"], x["known"]),
        reverse=True
    )

    best = candidates[0]
    return best, candidates


def map_location():
    print("\n=== STEP 4: MAP LOCATION ===")

    color_file = os.path.join(RESULTS_DIR, "local_color_3x3.txt")
    object_file = os.path.join(RESULTS_DIR, "local_object_3x3.txt")

    local_color_3x3 = read_local_3x3(color_file)
    local_object_3x3 = read_local_3x3(object_file)

    best, candidates = find_best_match(local_color_3x3, BIG_GRID)

    if best is None:
        raise RuntimeError("No valid match found in BIG_GRID.")

    camera_direction_before_scan = best["facing"]

    final_local_heading_after_scan, camera_direction_after_scan = get_final_camera_direction_after_scan(
        start_map_direction=camera_direction_before_scan,
        scan_start_local=SCAN_START_LOCAL,
        scan_sweep=SCAN_SWEEP,
        num_views=NUM_VIEWS
    )

    compact_17char = build_compact_17char(
        best["window"],
        local_object_3x3,
        camera_direction_after_scan
    )

    print(f"compact_17char = {compact_17char}")

    with open(FINAL_COMPACT_FILE, "w") as f:
        f.write(compact_17char + "\n")

    print(f"Saved final compact result to: {FINAL_COMPACT_FILE}")
    return compact_17char


# =========================================================
# MAIN RUNNER
# =========================================================

def main():
    ensure_dirs()

    try:
        capture_scan()
        detect_colors()
        detect_objects()
        compact_17char = map_location()

        print("\n=== DONE ===")
        print(f"Final compact_17char: {compact_17char}")

    except Exception as e:
        print(f"\nERROR: {e}")


if __name__ == "__main__":
    main()

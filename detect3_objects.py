import os
import json
import cv2
import numpy as np

# =========================================================
# CONFIG
# =========================================================

SCAN_DIR = "scan_images"
DEBUG_DIR = "debug_objects"
RESULTS_DIR = "results"

HEADINGS = ["front", "right", "back", "left"]

# ---------------------------------------------------------
# Updated from your screenshot values
# ---------------------------------------------------------

ROI_TOP_FRAC = 0.34
ROI_BOT_FRAC = 0.94

SLOT_PAD_X_FRAC = 0.03
SLOT_PAD_Y_FRAC = 0.06

# White / black / red thresholds from screenshot
WHITE_S_MAX = 184
WHITE_V_MIN = 170
MIN_WHITE_AREA = 2500

RED_H1_MAX = 24
RED_H2_MIN = 15
RED_S_MIN = 47
RED_V_MIN = 0

BLUE_H_MIN = 147
BLUE_H_MAX = 174
BLUE_S_MIN = 0
BLUE_V_MIN = 0

BLACK_V_MAX = 196

RED_RATIO_TH = 0.33
BLACK_RATIO_TH = 0.26
BLUE_RATIO_MAX = 0.78
RB_MARGIN = 1.30
TB_MARGIN = 1.30
WHITE_RATIO_MIN = 0.18

OPEN_K = 1   # screenshot showed 0, but kernel cannot be 0
CLOSE_K = 5

# Still keep these for line / X-shape support
CANNY1 = 40
CANNY2 = 120
HOUGH_TH = 30
MIN_LINE = 8
MAX_GAP = 40

BLUR_ODD = 1

# Optional small-blob cleanup
MIN_RED_BLOB_AREA = 120
MIN_BLUE_BLOB_AREA = 120
MIN_BLACK_BLOB_AREA = 120

# Object meaning:
# O = obstacle   (white box with thick red X)
# T = target     (white box with thick black X)
# E = empty
# ? = unknown

HEADING_TO_POSITIONS = {
    "front": [(-1, +1), (0, +1), (+1, +1)],
    "right": [(+1, +1), (+1, 0), (+1, -1)],
    "back":  [(+1, -1), (0, -1), (-1, -1)],
    "left":  [(-1, -1), (-1, 0), (-1, +1)],
}


def get_three_slot_rois(img):
    h, w = img.shape[:2]

    y0 = int(ROI_TOP_FRAC * h)
    y1 = int(ROI_BOT_FRAC * h)

    if y1 <= y0:
        return []

    band = img[y0:y1, :]
    bh, bw = band.shape[:2]

    slots = []

    for i in range(3):
        sx0 = int(i * bw / 3)
        sx1 = int((i + 1) * bw / 3)

        pad_x = int(SLOT_PAD_X_FRAC * (sx1 - sx0))
        pad_y = int(SLOT_PAD_Y_FRAC * bh)

        cx0 = max(0, sx0 + pad_x)
        cx1 = min(bw, sx1 - pad_x)
        cy0 = max(0, pad_y)
        cy1 = min(bh, bh - pad_y)

        crop = band[cy0:cy1, cx0:cx1]
        slots.append(crop)

    return slots


def matrix_rows_from_grid(final_grid):
    rows = []
    for row in [1, 0, -1]:
        vals = []
        for col in [-1, 0, 1]:
            vals.append(final_grid.get((col, row), "?"))
        rows.append(vals)
    return rows


def pretty_print_matrix(final_grid):
    rows = matrix_rows_from_grid(final_grid)
    for row in rows:
        print(" ".join(row))


def save_matrix_txt(path, final_grid):
    rows = matrix_rows_from_grid(final_grid)
    with open(path, "w") as f:
        for row in rows:
            f.write(" ".join(row) + "\n")


def remove_small_blobs(mask, min_area):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask)
    for c in cnts:
        if cv2.contourArea(c) >= min_area:
            cv2.drawContours(cleaned, [c], -1, 255, thickness=cv2.FILLED)
    return cleaned


def clean_mask(mask, open_k=1, close_k=5):
    if open_k < 1:
        open_k = 1
    if close_k < 1:
        close_k = 1

    if open_k % 2 == 0:
        open_k += 1
    if close_k % 2 == 0:
        close_k += 1

    kernel_open = np.ones((open_k, open_k), np.uint8)
    kernel_close = np.ones((close_k, close_k), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    return mask


def detect_one_object_slot(slot_bgr):
    if slot_bgr is None or slot_bgr.size == 0:
        return "?", {}

    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2GRAY)

    if BLUR_ODD > 1:
        k = BLUR_ODD if BLUR_ODD % 2 == 1 else BLUR_ODD + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    # -----------------------------------------------------
    # White mask (HSV-based from screenshot)
    # -----------------------------------------------------
    white_mask = cv2.inRange(
        hsv,
        np.array([0, 0, WHITE_V_MIN], dtype=np.uint8),
        np.array([180, WHITE_S_MAX, 255], dtype=np.uint8)
    )
    white_mask = clean_mask(white_mask, OPEN_K, CLOSE_K)

    # -----------------------------------------------------
    # Red mask
    # -----------------------------------------------------
    lower_red1 = np.array([0, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper_red1 = np.array([RED_H1_MAX, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([RED_H2_MIN, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red1, red2)
    red_mask = clean_mask(red_mask, OPEN_K, CLOSE_K)
    red_mask = remove_small_blobs(red_mask, MIN_RED_BLOB_AREA)

    # -----------------------------------------------------
    # Blue mask
    # -----------------------------------------------------
    if BLUE_H_MIN <= BLUE_H_MAX:
        blue_mask = cv2.inRange(
            hsv,
            np.array([BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN], dtype=np.uint8),
            np.array([BLUE_H_MAX, 255, 255], dtype=np.uint8)
        )
    else:
        b1 = cv2.inRange(
            hsv,
            np.array([BLUE_H_MIN, BLUE_S_MIN, BLUE_V_MIN], dtype=np.uint8),
            np.array([179, 255, 255], dtype=np.uint8)
        )
        b2 = cv2.inRange(
            hsv,
            np.array([0, BLUE_S_MIN, BLUE_V_MIN], dtype=np.uint8),
            np.array([BLUE_H_MAX, 255, 255], dtype=np.uint8)
        )
        blue_mask = cv2.bitwise_or(b1, b2)

    blue_mask = clean_mask(blue_mask, OPEN_K, CLOSE_K)
    blue_mask = remove_small_blobs(blue_mask, MIN_BLUE_BLOB_AREA)

    # -----------------------------------------------------
    # Black mask
    # -----------------------------------------------------
    v = hsv[:, :, 2]
    black_mask = np.where(v <= BLACK_V_MAX, 255, 0).astype(np.uint8)
    black_mask = clean_mask(black_mask, OPEN_K, CLOSE_K)
    black_mask = remove_small_blobs(black_mask, MIN_BLACK_BLOB_AREA)

    # -----------------------------------------------------
    # Ratios
    # -----------------------------------------------------
    total = float(slot_bgr.shape[0] * slot_bgr.shape[1]) + 1e-6
    white_ratio = float(np.count_nonzero(white_mask)) / total
    red_ratio = float(np.count_nonzero(red_mask)) / total
    blue_ratio = float(np.count_nonzero(blue_mask)) / total
    black_ratio = float(np.count_nonzero(black_mask)) / total

    # -----------------------------------------------------
    # X-shape support from edges
    # -----------------------------------------------------
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
        "white_ratio": round(white_ratio, 4),
        "red_ratio": round(red_ratio, 4),
        "blue_ratio": round(blue_ratio, 4),
        "black_ratio": round(black_ratio, 4),
        "diag_pos": int(diag_pos),
        "diag_neg": int(diag_neg),
        "has_x_shape": bool(has_x_shape),
    }

    # -----------------------------------------------------
    # Decision logic using screenshot thresholds
    # -----------------------------------------------------
    obstacle_ok = (
        white_ratio >= WHITE_RATIO_MIN and
        red_ratio >= RED_RATIO_TH and
        blue_ratio <= BLUE_RATIO_MAX and
        red_ratio > blue_ratio * RB_MARGIN and
        has_x_shape
    )

    target_ok = (
        white_ratio >= WHITE_RATIO_MIN and
        black_ratio >= BLACK_RATIO_TH and
        black_ratio > blue_ratio * TB_MARGIN and
        has_x_shape
    )

    if obstacle_ok and not target_ok:
        return "O", metrics

    if target_ok and not obstacle_ok:
        return "T", metrics

    if obstacle_ok and target_ok:
        if red_ratio > black_ratio:
            return "O", metrics
        else:
            return "T", metrics

    # If nothing strong enough, call empty
    if white_ratio < WHITE_RATIO_MIN and red_ratio < RED_RATIO_TH and black_ratio < BLACK_RATIO_TH:
        return "E", metrics

    return "?", metrics


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

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
            print(f"ERROR: Missing image: {path}")
            return

        img = cv2.imread(path)
        if img is None:
            print(f"ERROR: Could not read image: {path}")
            return

        slots = get_three_slot_rois(img)
        if len(slots) != 3:
            print(f"ERROR: Could not build 3 slots for heading: {heading}")
            return

        heading_info = []
        print(f"\nHeading: {heading}")

        for i, tile in enumerate(slots):
            dbg_name = os.path.join(DEBUG_DIR, f"{heading}_slot{i}.jpg")
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

    print("\nFinal 3x3 object matrix:")
    pretty_print_matrix(final_grid)

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

    print(f"\nSaved: {json_path}")
    print(f"Saved: {txt_path}")
    print(f"Saved debug crops in: {DEBUG_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()

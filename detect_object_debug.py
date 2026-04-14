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

ROI_TOP_FRAC = 0.34
ROI_BOT_FRAC = 0.94

SLOT_PAD_X_FRAC = 0.03
SLOT_PAD_Y_FRAC = 0.06

WHITE_MIN = 150
BLACK_MAX = 80
RED_S_MIN = 100
RED_V_MIN = 100

BLUR_ODD = 3

# candidate size relative to slot size
MIN_BOX_AREA_FRAC = 0.04
MAX_BOX_AREA_FRAC = 0.22

MIN_BOX_W_FRAC = 0.18
MAX_BOX_W_FRAC = 0.55
MIN_BOX_H_FRAC = 0.18
MAX_BOX_H_FRAC = 0.55

MIN_SOLIDITY = 0.75
MIN_EXTENT = 0.55
MIN_ASPECT = 0.60
MAX_ASPECT = 1.40

# inner region checks
INNER_MARGIN_FRAC = 0.12
RED_RATIO_TH = 0.08
BLACK_RATIO_TH = 0.08

# X detection
CANNY1 = 50
CANNY2 = 140
HOUGH_TH = 18
MIN_LINE = 10
MAX_GAP = 12

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


def ensure_odd(k):
    return k if k % 2 == 1 else k + 1


def detect_x_lines(binary_img):
    edges = cv2.Canny(binary_img, CANNY1, CANNY2)

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

    has_x = (diag_pos >= 1 and diag_neg >= 1)
    return has_x, diag_pos, diag_neg, edges


def find_best_box_candidate(slot_bgr):
    h, w = slot_bgr.shape[:2]
    slot_area = h * w

    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2GRAY)

    if BLUR_ODD > 1:
        gray = cv2.GaussianBlur(gray, (ensure_odd(BLUR_ODD), ensure_odd(BLUR_ODD)), 0)

    # bright/low-sat white candidate
    white_mask = cv2.inRange(hsv, np.array([0, 0, WHITE_MIN], dtype=np.uint8),
                                  np.array([180, 80, 255], dtype=np.uint8))

    kernel = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1.0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 40:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        rect_area = bw * bh
        if rect_area <= 0:
            continue

        area_frac = area / float(slot_area)
        w_frac = bw / float(w)
        h_frac = bh / float(h)
        aspect = bw / float(bh) if bh > 0 else 999

        if not (MIN_BOX_AREA_FRAC <= area_frac <= MAX_BOX_AREA_FRAC):
            continue
        if not (MIN_BOX_W_FRAC <= w_frac <= MAX_BOX_W_FRAC):
            continue
        if not (MIN_BOX_H_FRAC <= h_frac <= MAX_BOX_H_FRAC):
            continue
        if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        extent = area / rect_area if rect_area > 0 else 0.0

        if solidity < MIN_SOLIDITY or extent < MIN_EXTENT:
            continue

        # prefer big, centered candidates
        cx = x + bw / 2.0
        cy = y + bh / 2.0
        center_dx = abs(cx - w / 2.0) / max(1.0, w / 2.0)
        center_dy = abs(cy - h / 2.0) / max(1.0, h / 2.0)
        center_penalty = center_dx + center_dy

        score = (2.0 * area_frac) + solidity + extent - 0.5 * center_penalty

        if score > best_score:
            best_score = score
            best = {
                "rect": (x, y, bw, bh),
                "area": area,
                "area_frac": area_frac,
                "aspect": aspect,
                "solidity": solidity,
                "extent": extent,
                "mask": white_mask
            }

    return best, white_mask


def detect_one_object_slot(slot_bgr):
    if slot_bgr is None or slot_bgr.size == 0:
        return "?", {}

    h, w = slot_bgr.shape[:2]
    gray = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(slot_bgr, cv2.COLOR_BGR2HSV)

    best, white_mask = find_best_box_candidate(slot_bgr)

    if best is None:
        empty_bright_ratio = float(np.count_nonzero(gray > WHITE_MIN)) / gray.size
        metrics = {
            "candidate_found": False,
            "slot_bright_ratio": round(empty_bright_ratio, 4),
        }
        if empty_bright_ratio < 0.10:
            return "E", metrics
        return "?", metrics

    x, y, bw, bh = best["rect"]

    mx = int(INNER_MARGIN_FRAC * bw)
    my = int(INNER_MARGIN_FRAC * bh)

    ix0 = max(0, x + mx)
    iy0 = max(0, y + my)
    ix1 = min(w, x + bw - mx)
    iy1 = min(h, y + bh - my)

    if ix1 <= ix0 or iy1 <= iy0:
        return "?", {
            "candidate_found": True,
            "reason": "inner_roi_invalid"
        }

    inner = slot_bgr[iy0:iy1, ix0:ix1]
    inner_hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    inner_gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)

    lower_red1 = np.array([0, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, RED_S_MIN, RED_V_MIN], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    red1 = cv2.inRange(inner_hsv, lower_red1, upper_red1)
    red2 = cv2.inRange(inner_hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red1, red2)

    black_mask = cv2.inRange(inner_gray, 0, BLACK_MAX)

    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN, kernel)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel)

    red_ratio = float(np.count_nonzero(red_mask)) / red_mask.size
    black_ratio = float(np.count_nonzero(black_mask)) / black_mask.size

    has_red_x, red_diag_pos, red_diag_neg, red_edges = detect_x_lines(red_mask)
    has_black_x, black_diag_pos, black_diag_neg, black_edges = detect_x_lines(black_mask)

    metrics = {
        "candidate_found": True,
        "box_x": int(x),
        "box_y": int(y),
        "box_w": int(bw),
        "box_h": int(bh),
        "box_area_frac": round(best["area_frac"], 4),
        "aspect": round(best["aspect"], 4),
        "solidity": round(best["solidity"], 4),
        "extent": round(best["extent"], 4),
        "red_ratio": round(red_ratio, 4),
        "black_ratio": round(black_ratio, 4),
        "has_red_x": bool(has_red_x),
        "has_black_x": bool(has_black_x),
        "red_diag_pos": int(red_diag_pos),
        "red_diag_neg": int(red_diag_neg),
        "black_diag_pos": int(black_diag_pos),
        "black_diag_neg": int(black_diag_neg),
    }

    if red_ratio >= RED_RATIO_TH and has_red_x:
        return "O", metrics

    if black_ratio >= BLACK_RATIO_TH and has_black_x:
        return "T", metrics

    return "?", metrics


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    final_grid = {
        (-1, +1): "?",
        (0, +1): "?",
        (+1, +1): "?",
        (-1, 0): "?",
        (0, 0): "A",
        (+1, 0): "?",
        (-1, -1): "?",
        (0, -1): "?",
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

import os
import sys
import subprocess

# =========================================================
# CONFIG
# =========================================================

SCAN_DIR = "scan_images"
RESULTS_DIR = "results"

CAPTURE_SCRIPT = "capture_scan.py"
COLOR_SCRIPT = "detect_colors.py"
OBJECT_SCRIPT = "detect_objects.py"   # safe ArUco version
MAP_SCRIPT = "map_location.py"

HEADINGS = ["front", "right", "back", "left"]

os.makedirs(SCAN_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================

def run_step(script_name, step_name, required=True):
    print(f"\n{step_name}")

    if not os.path.exists(script_name):
        msg = f"[WARN] {script_name} not found."
        print(msg)

        if required:
            raise FileNotFoundError(msg)

        return False

    print(f"Running {script_name}...")

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=False
        )

        if result.returncode != 0:
            msg = f"[ERROR] {script_name} failed with return code {result.returncode}"
            print(msg)

            if required:
                raise RuntimeError(msg)

            return False

        print(f"{script_name} completed.")
        return True

    except Exception as e:
        msg = f"[ERROR] Could not run {script_name}: {e}"
        print(msg)

        if required:
            raise

        return False


def check_scan_images():
    print("\nChecking scan images...")

    missing = []

    for heading in HEADINGS:
        path = os.path.join(SCAN_DIR, f"{heading}.jpg")

        if not os.path.exists(path):
            missing.append(path)

    if missing:
        print("[ERROR] Missing scan images:")
        for path in missing:
            print("  ", path)

        raise FileNotFoundError("Missing one or more scan images.")

    print("All scan images found.")


def check_file(path, description, required=True):
    if os.path.exists(path):
        print(f"[OK] {description}: {path}")
        return True

    print(f"[WARN] Missing {description}: {path}")

    if required:
        raise FileNotFoundError(path)

    return False


# =========================================================
# MAIN RUNNER
# =========================================================

def main():
    print("===================================")
    print("PI SENSING RUNNER")
    print("SAFE ARUCO OBJECT DETECTION")
    print("===================================")

    # STEP 1: capture four saved pictures
    run_step(CAPTURE_SCRIPT, "STEP 1: CAPTURE SCAN", required=False)

    # Make sure front/right/back/left images exist
    check_scan_images()

    # STEP 2: detect tile colors
    run_step(COLOR_SCRIPT, "STEP 2: DETECT COLORS", required=True)

    # Optional expected color output check
    check_file(
        os.path.join(RESULTS_DIR, "color_3x3.txt"),
        "color grid",
        required=False
    )

    # STEP 3: detect objects using safe ArUco script
    # This runs in a separate Python process for better Pi stability.
    run_step(OBJECT_SCRIPT, "STEP 3: DETECT OBJECTS WITH SAFE ARUCO", required=True)

    # Required object output
    check_file(
        os.path.join(RESULTS_DIR, "object_3x3.txt"),
        "object grid",
        required=True
    )

    # STEP 4: map location using color + object outputs
    run_step(MAP_SCRIPT, "STEP 4: MAP LOCATION / ORIENTATION / FINAL OUTPUT", required=True)

    print("\n===================================")
    print("RUNNER FINISHED")
    print("===================================")


if __name__ == "__main__":
    main()

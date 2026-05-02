"""
Invisible Cloak - OpenCV Project
=================================
Step 1: Run the script
Step 2: A window opens — type a color name and press Enter
Step 3: Hold that colored object in front of the camera
Step 4: Watch it disappear! Press 'q' to quit.

Supported colors: red, green, blue, yellow, orange, purple, pink, cyan, white
"""

import cv2
import numpy as np
import time

# ── HSV color ranges ──────────────────────────────────────────────────────────
COLOR_RANGES = {
    "red":    [([0, 120, 70],  [10, 255, 255]),
               ([170, 120, 70], [180, 255, 255])],   # red wraps around 0°/180°
    "green":  [([36, 80, 40],  [86, 255, 255])],
    "blue":   [([94, 80, 40],  [126, 255, 255])],
    "yellow": [([20, 100, 100],[35, 255, 255])],
    "orange": [([10, 100, 100],[20, 255, 255])],
    "purple": [([125, 50, 50], [160, 255, 255])],
    "pink":   [([145, 50, 50], [170, 255, 255])],
    "cyan":   [([78, 80, 40],  [100, 255, 255])],
    "white":  [([0, 0, 200],   [180, 30, 255])],
}

def get_mask(hsv_frame, color_name):
    """Build a binary mask for the chosen color."""
    ranges = COLOR_RANGES.get(color_name)
    if not ranges:
        return None
    mask = np.zeros(hsv_frame.shape[:2], dtype=np.uint8)
    for (lo, hi) in ranges:
        mask |= cv2.inRange(hsv_frame, np.array(lo), np.array(hi))

    # Clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
    return mask

def ask_color_in_window(cap):
    """
    Show a simple text-input window where the user types a color.
    Returns the chosen color string (lowercase), or None if cancelled.
    """
    supported = sorted(COLOR_RANGES.keys())
    typed = ""

    while True:
        canvas = np.zeros((420, 640, 3), dtype=np.uint8)

        # Title
        cv2.putText(canvas, "Invisible Cloak", (140, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 255), 3)

        # Instructions
        cv2.putText(canvas, "Type a color and press ENTER:", (60, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        # Supported colors list
        line = "  ".join(supported)
        cv2.putText(canvas, line, (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

        # Input box
        cv2.rectangle(canvas, (60, 190), (580, 250), (50, 50, 50), -1)
        cv2.rectangle(canvas, (60, 190), (580, 250), (0, 200, 200), 2)
        cv2.putText(canvas, typed + "|", (75, 233),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Feedback
        if typed and typed not in supported:
            cv2.putText(canvas, "Unknown color — check spelling",
                        (80, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
        elif typed in supported:
            cv2.putText(canvas, f"Ready! Press ENTER to use '{typed}'",
                        (80, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)

        cv2.putText(canvas, "Press ESC to quit",
                    (220, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

        cv2.imshow("Invisible Cloak - Color Picker", canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:          # ESC → quit
            return None
        elif key == 13:        # ENTER → confirm
            if typed in supported:
                return typed
        elif key == 8 or key == 127:   # BACKSPACE
            typed = typed[:-1]
        elif key != 255:
            ch = chr(key).lower()
            if ch.isalpha() and len(typed) < 12:
                typed += ch

def flush_keys(duration=0.6):
    """Drain stale keypresses for `duration` seconds so the main loop starts clean."""
    end = time.time() + duration
    while time.time() < end:
        cv2.waitKey(30)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    # Warm up by reading frames — no time.sleep needed
    print("Warming up camera…")
    bg_frames = []
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            bg_frames.append(frame)
    background = np.median(bg_frames, axis=0).astype(np.uint8)

    # ── Ask user for a color ─────────────────────────────────────────────────
    chosen_color = ask_color_in_window(cap)
    cv2.destroyAllWindows()

    if chosen_color is None:
        cap.release()
        return

    print(f"[Invisible Cloak] Color selected: {chosen_color}")
    print("Hold the object in front of the camera.  Press 'q' or ESC to quit.")

    win_name = f"Invisible Cloak  [{chosen_color}]  |  q = quit"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    # KEY FIX: flush any stale keypresses before entering the main loop
    flush_keys(duration=0.6)

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = get_mask(hsv, chosen_color)
        if mask is None:
            break

        mask_inv = cv2.bitwise_not(mask)

        bg_part = cv2.bitwise_and(background, background, mask=mask)
        fg_part = cv2.bitwise_and(frame, frame, mask=mask_inv)
        output  = cv2.addWeighted(bg_part, 1, fg_part, 1, 0)

        cv2.imshow(win_name, output)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()

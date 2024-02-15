import cv2
import numpy as np


def draw_midline_between_parallel_lines(frame, lines, region_of_interest_vertices):
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Assuming the two lines are the first two in the list and calculating midpoints
        if len(lines) >= 2:
            # Calculate midpoint of the start and end points of the first line
            x1_mid = int((lines[0][0][0] + lines[1][0][0]) / 2)
            y1_mid = int((lines[0][0][1] + lines[1][0][1]) / 2)
            # Calculate midpoint of the start and end points of the second line
            x2_mid = int((lines[0][0][2] + lines[1][0][2]) / 2)
            y2_mid = int((lines[0][0][3] + lines[1][0][3]) / 2)
            # Draw the midline
            cv2.line(frame, (x1_mid, y1_mid), (x2_mid, y2_mid), (255, 0, 0), 3)


def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


def process_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray, 100, 120)
    rows, cols = img.shape[:2]
    # Adjusted to make the region more square-like
    square_size = min(cols, rows) * 0.4  
    center_col, center_row = cols // 2, rows // 2
    half_square = square_size // 2
    bottom_left = [center_col - half_square, center_row + half_square]
    top_left = [center_col - half_square, center_row - half_square]
    bottom_right = [center_col + half_square, center_row + half_square]
    top_right = [center_col + half_square, center_row - half_square]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    roi_image = region_of_interest(canny_image, vertices)
    lines = cv2.HoughLinesP(roi_image, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)
    draw_midline_between_parallel_lines(img, lines, vertices)
    cv2.polylines(img, vertices, isClosed=True, color=(255, 0, 0), thickness=5)
    return img


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = process_image(frame)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

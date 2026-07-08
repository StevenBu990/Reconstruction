import cv2

# Load inputs
img = cv2.imread("view3Input.png")
mask = cv2.imread("GT_HandleMask.png", 0)  # grayscale mask

# Safety check (VERY important)
if img is None:
    raise ValueError("Image not found")
if mask is None:
    raise ValueError("Mask not found")

# Run geometry-only reconstruction (inpainting)
result = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

# Save output
cv2.imwrite("geometryView3.png", result)

print("Done: geometry-only reconstruction saved")
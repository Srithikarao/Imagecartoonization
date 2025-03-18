import cv2
import numpy as np

# Load and resize image
image = cv2.imread("viratkohli.jpg")  # Replace with your image path
image = cv2.resize(image, (600, 600))
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

def neon_glow(image):
    edges = cv2.Canny(image, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 0.7, edges_colored, 0.3, 0)

def stylized_image(image):
    return cv2.stylization(image, sigma_s=150, sigma_r=0.25)

def cartoonize(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    color = cv2.edgePreservingFilter(image, flags=2, sigma_s=150, sigma_r=0.3)
    edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, blockSize=9, C=2)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    data = np.float32(image).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 9
    _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()].reshape(image.shape)
    return cv2.bitwise_and(segmented_image, edges_colored)



filters = {
    "Neon Glow": neon_glow(image),
    "Stylized Image": stylized_image(image),
    "Cartoonized Image": cartoonize(image)
   
}

for name, result in filters.items():
    cv2.imshow(name, result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Save outputs
for name, result in filters.items():
    cv2.imwrite(f"{name.lower().replace(' ', '_')}.jpg", result)

print("Images saved successfully!")

























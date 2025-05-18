# Find an image and try to label a bounding box that contains the object.



import cv2

# Load the image
image = cv2.imread('black_apples.jpeg')

# Define the bounding box for the center apple (x_min, y_min, x_max, y_max)
# These values are estimated for the center apple in your image

y_min = 40
y_max = 310

x_min = 10
x_width =320

for i in range(1, 4):
    if i>1:
        x_min = x_min + x_width + 20
    x_max = x_min + x_width -25

    # Draw the bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color=(0, 255, 0), thickness=2)

    # Add the label above the bounding box
    label = "black apple"
    cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8, color=(0, 255, 0), thickness=2)

# Save or display the result
cv2.imwrite('labeled_black_apple.jpg', image)
# To display the image (optional):
# cv2.imshow('Labeled Image', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

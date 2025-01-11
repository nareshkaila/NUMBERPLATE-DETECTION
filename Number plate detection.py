import cv2
import imutils
import pytesseract

# Set the path to your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r'/Users/kailanaresh/Downloads/A VS CODE/28. Number plate detection/jeep.jpg'  # Update this with the correct path

# Load the image
image_path = '/Users/kailanaresh/Downloads/A VS CODE/28. Number plate detection/jeep.jpg'  # Update this path as needed
image = cv2.imread(image_path)

if image is None:
    print("Error: Image not found!")
    exit()

# Resize the image for better processing (optional)
resized_image = imutils.resize(image)
cv2.imshow('Original Image', resized_image)
cv2.waitKey(0)

# Convert the image to grayscale
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Image", gray_image)
cv2.waitKey(0)

# Apply bilateral filter to reduce noise while preserving edges
gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
cv2.imshow("Smoothened Image", gray_image)
cv2.waitKey(0)

# Use Canny edge detection to detect edges in the image
edged = cv2.Canny(gray_image, 30, 200)
cv2.imshow("Edge Detection", edged)
cv2.waitKey(0)

# Find contours in the edge-detected image
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Draw all contours on a copy of the original image
image_contours = resized_image.copy()
cv2.drawContours(image_contours, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Contours", image_contours)
cv2.waitKey(0)

# Sort the contours based on area and keep the top 30 largest ones
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

# Initialize the screenCnt (contour that represents the license plate)
screenCnt = None
image_top_cnts = resized_image.copy()
cv2.drawContours(image_top_cnts, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Top 30 Contours", image_top_cnts)
cv2.waitKey(0)

# Loop over the contours to find the rectangle that represents the license plate
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)

    # If the contour has 4 points, we consider it as the license plate
    if len(approx) == 4:
        screenCnt = approx
        x, y, w, h = cv2.boundingRect(c)
        new_img = resized_image[y:y+h, x:x+w]
        cv2.imwrite(f'./license_plate_{x}.png', new_img)  # Save the detected license plate image
        cv2.imshow("Detected License Plate", new_img)
        cv2.waitKey(0)
        
        # Use pytesseract to extract the text from the license plate
        license_text = pytesseract.image_to_string(new_img, config='--psm 8')
        print("Detected License Plate Text:", license_text)
        
        break

# If no contour was found, display a message
if screenCnt is None:
    print("No license plate detected.")

# Draw the final detected license plate contour on the original image
if screenCnt is not None:
    cv2.drawContours(resized_image, [screenCnt], -1, (0, 255, 0), 3)
    cv2.imshow("License Plate Detection", resized_image)
    cv2.waitKey(0)

# Clean up and close all windows
cv2.destroyAllWindows()

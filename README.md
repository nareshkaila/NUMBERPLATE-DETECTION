# NUMBERPLATE-DETECTION
This project demonstrates how to detect and extract text from vehicle license plates using computer vision techniques. It leverages OpenCV for image processing and Tesseract OCR to recognize the text on the detected number plates. This approach is useful for applications such as automatic number plate recognition (ANPR) systems.

#Technologies Used
Python: Programming language used to build the project.
OpenCV: Used for image processing (e.g., resizing, edge detection, contour finding).
Tesseract OCR: Used for optical character recognition to extract text from the detected license plate.
imutils: A library for simplifying image manipulation tasks like resizing.

#Key Features
License Plate Detection: The script detects and highlights the license plate from an input image.
Text Extraction: Uses Tesseract OCR to extract the alphanumeric characters from the detected license plate.
Edge Detection & Contour Finding: Applies Canny edge detection and contour finding to locate potential license plates.
Image Preprocessing: Includes various preprocessing techniques like grayscale conversion, bilateral filtering, and smoothing for better text extraction accuracy.

#Output

Original Image: The input image.
Gray Image: The grayscale version of the image.
Smoothened Image: The image after applying bilateral filtering.
Edge Detected Image: The image after applying Canny edge detection.
Contours: The contours of the detected objects.
Detected License Plate: The final detected license plate region with extracted text.

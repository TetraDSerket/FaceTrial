import cv2
import sys

# Get user supplied values
imagePath = sys.argv[1] #command line arguments, takes in user input of image
cascPath = sys.argv[2] #path to the default face-detection of OpenCV

# Create the haar cascade with the face cascade path
# The cascade is an XML file that contains how to detect faces
faceCascade = cv2.CascadeClassifier(cascPath) 

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert image to grayscale

# Detect faces in the image
# detectMultiScale detects objects of the type of cascade you're calling it with
# scalefactor deals with that fact that some faces are larger than others
# The detection algorithm uses a moving window to detect objects
# minNeighbors defines how many objects are detected near the current one before it declares the face found
# minSize gives the size of each window.

# returns a list of rectangles where it thinks there's a face
faces = faceCascade.detectMultiScale(  
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

# prints number of faces found
print "Found {0} faces!".format(len(faces)) 

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)

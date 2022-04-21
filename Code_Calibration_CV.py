#####################################################################################################################################
#	PROGRAM BY : MC IATRIDES
#	LAST UPDATE : 21-04-2022
#	TITLE : Exercise #1 - (25-04-2022)
#   SUBTITLE : Camera Calibration using Checker Pattern
#	REDACTED FOR : COMPUTER VISION
#####################################################################################################################################

##### PACKAGES ######################################################################################################################
from numpy import *
import cv2 as cv
import glob
#####################################################################################################################################

###### ANALYSIS PART ################################################################################################################
print('START TESTS')

#Setup
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = zeros((7*11,3), float32)
objp[:,:2] = mgrid[0:11,0:7].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.pix
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('images/*.jpg')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (11,7), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (11,7), corners2, ret)
        #cv.imshow('img', img)
        cv.imwrite('output.jpg',img)
        #cv.waitKey(500)

cv.destroyAllWindows()

#Calibration
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#Undistortion
img = cv.imread('images/20220421_171148.jpg')
h,  w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
# undistort
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('calibresult.png', dst)

#Re-projection Error
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error
print( "total error: {}".format(mean_error/len(objpoints)) )

#Intrinsic matrix
print("intrinsic matric :\n", mtx)

#Focal length
focal = mtx[0,0]
focal *= 1.4e-3
print('focal length (mm): ',focal)

    
print('END TESTS')
#####################################################################################################################################

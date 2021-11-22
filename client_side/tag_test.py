import numpy as np
import cv2 as cv
import apriltag
import socket
from time import perf_counter
from datetime import datetime
import csv
from os import system
import matplotlib.pyplot as plt

#CONVERSION FACTOR: 500 STEPS = 31mm

# Create a black image
# Draw a diagonal blue line with thickness of 5 px

real_points = np.array([
    [0, 0],
    [620, 0],
    [620, 300],
    [0, 300]
], np.float32)

cap = cv.VideoCapture("/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM-video-index0")  # Opens the webcam
w, h = cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT)
print(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))

points_dataset = np.array(["Time","x","y","theta"])


def bounding_box(results, image, real_points, points_dataset):
    tag_mat = []
    tracker_tag_found = False

   # print(len(results))
    if len(results) == 4:

        for marker in results:
            print(marker.tag_id)

            if marker.tag_id <= 3:
                tag_mat.append([marker.tag_id, int(marker.corners[marker.tag_id][0]), int(marker.corners[marker.tag_id][1])])

            else:
                pass


        tag_mat = np.array(tag_mat)
        tag_mat = tag_mat[tag_mat[:, 0].argsort()]

        #print(tag_mat[:,0])

        ids = list(tag_mat[:,0])
        if ids == [0, 1, 2, 3]:

            x_points = tag_mat[:, 1]
            y_points = tag_mat[:, 2]
            points = zip(x_points, y_points)
            points = np.array(list(points), np.float32)
            #print(points)

            cv.line(image, (x_points[0], y_points[0]), (x_points[1], y_points[1]), (0, 255, 0), 2)
            cv.line(image, (x_points[1], y_points[1]), (x_points[2], y_points[2]), (0, 255, 0), 2)
            cv.line(image, (x_points[2], y_points[2]), (x_points[3], y_points[3]), (0, 255, 0), 2)
            cv.line(image, (x_points[3], y_points[3]), (x_points[0], y_points[0]), (0, 255, 0), 2)


        else:
            pass
    else:
        pass

    cv.imshow("Image", image)

    return points_dataset

if __name__ ==  "__main__":

    if not cap.isOpened():  # Sometimes cap doesn't inisliatlies capture
        print("Cannot open camera")
        exit()


    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    while True:

        # Capture frame-by-frame
        ret, image = cap.read()  # Ret is a bool that is true if the frame is read correclty
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        gray = cv.undistort(gray, mtx, dist, None
                            , newcameramtx)
        options = apriltag.DetectorOptions(families="tag16h5")
        detector = apriltag.Detector(options)
        results = detector.detect(gray)
        # Display the resulting frame
        points_dataset = bounding_box(results, image, real_points, points_dataset)
        print(points_dataset[-1])
        if cv.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

    points_dataset = np.c_[points_dataset]

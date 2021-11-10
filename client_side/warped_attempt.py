import numpy as np
import cv2 as cv
import apriltag

# Create a black image
# Draw a diagonal blue line with thickness of 5 px

real_points = np.array([
    [0, 0],
    [620, 0],
    [620, 300],
    [0, 300]
], np.float32)

cap = cv.VideoCapture(0)  # Opens the webcam

print(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))


def bounding_box(results, image, real_points):
    tag_mat = []
    i = 0
    if len(results) == 5:
        for marker in results:
            tag_mat.append([marker.tag_id, int(marker.center[0]), int(marker.center[1])])
            if marker.tag_id == 4:
                pA, pB, pC, pD = marker.corners
                angle_vect = np.array([int(pA[0]) - int(pB[0]), int(pA[1]) - int(pB[1]), 1])

        tag_mat = np.array(tag_mat)
        tag_mat = tag_mat[tag_mat[:, 0].argsort()]
        x_points = tag_mat[:, 1]
        y_points = tag_mat[:, 2]
        points = zip(x_points, y_points)
        points = np.array(list(points), np.float32)
        print(points)

        cv.line(image, (x_points[0], y_points[0]), (x_points[1], y_points[1]), (0, 255, 0), 2)
        cv.line(image, (x_points[1], y_points[1]), (x_points[2], y_points[2]), (0, 255, 0), 2)
        cv.line(image, (x_points[2], y_points[2]), (x_points[3], y_points[3]), (0, 255, 0), 2)
        cv.line(image, (x_points[3], y_points[3]), (x_points[0], y_points[0]), (0, 255, 0), 2)

        M = cv.getPerspectiveTransform(points[:-1], real_points)

        tranformed_angle_vect = np.matmul(M, np.transpose(angle_vect))

        warped = cv.warpPerspective(image, M, (620, 300))


        cv.circle(warped, (int(center_targer[0]), int(center_targer[0])), 5, (0, 255, 0), thickness=2)

        cv.imshow("warped", warped)

    else:
        pass

    cv.imshow("Image", image)


if not cap.isOpened():  # Sometimes cap doesn't inisliatlies capture
    print("Cannot open camera")
    exit()
while True:

    # Capture frame-by-frame
    ret, image = cap.read()  # Ret is a bool that is true if the frame is read correclty
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    options = apriltag.DetectorOptions(families="tag16h5")
    detector = apriltag.Detector(options)
    results = detector.detect(gray)
    # Display the resulting frame
    bounding_box(results, image, real_points)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

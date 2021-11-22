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

file_name = datetime.now().strftime("%d.%m-%H:%M:%S")
full_file_name = "data/run@" + file_name + ".csv"

csv_file = open(full_file_name, 'w')  # Create csv file to store the data collected
writer = csv.writer(csv_file)

#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.connect(('192.168.4.1',80))

# Create a black image
# Draw a diagonal blue line with thickness of 5 px

real_points = np.array([
    [0, 0],
    [680, 0],
    [680, 1080],
    [0, 1080]
], np.float32)

mtx = np.array([[682.86593085,   0.,         333.88762366],
 [  0.,         683.35533608, 194.30626825],
 [  0.,           0.,           1.        ]]

)

dst = np.array([[-4.52580443e-01,  3.54693558e-01,  6.31975682e-04,  3.27884916e-04,
  -5.29398897e-01]]

)

#cap = cv.VideoCapture(0)  # Opens the webcam
cap = cv.VideoCapture("/dev/v4l/by-id/usb-GENERAL_GENERAL_WEBCAM-video-index0")  # Opens the webcam
cap.set(cv.CAP_PROP_AUTO_EXPOSURE, 0.25)
cap.set(cv.CAP_PROP_EXPOSURE,100)

w, h = int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))


print(cap.get(cv.CAP_PROP_FRAME_WIDTH), cap.get(cv.CAP_PROP_FRAME_HEIGHT))

vid_file_name = 'data/' + file_name + '.mp4'

camerawriter= cv.VideoWriter(vid_file_name, cv.VideoWriter_fourcc(*'DIVX'), 20, (w,h))

points_dataset = np.array(["Time","x","y","theta"])


def bounding_box(results, image, real_points, points_dataset):
    tag_mat = []
    tracker_tag_found = False

   # print(len(results))
    if len(results) == 5:

        for marker in results:
            print(marker.tag_id)

            if marker.tag_id <= 3:
                tag_mat.append([marker.tag_id, int(marker.corners[marker.tag_id][0]), int(marker.corners[marker.tag_id][1])])
            elif marker.tag_id == 4:
                tag_mat.append([marker.tag_id,int(marker.center[0]),int(marker.center[1])])
                pA, pB, pC, pD = marker.corners
                angle_vect = np.array([[int(pA[0]), int(pA[1]), 1], [int(pB[0]), int(pB[1]), 1]])
                center_point = (int(marker.center[0]),int(marker.center[1]))

            else:
                pass


        tag_mat = np.array(tag_mat)
        tag_mat = tag_mat[tag_mat[:, 0].argsort()]

        #print(tag_mat[:,0])

        ids = list(tag_mat[:,0])
        if ids == [0, 1, 2, 3, 4]:

            x_points = tag_mat[:, 1]
            y_points = tag_mat[:, 2]
            points = zip(x_points, y_points)
            points = np.array(list(points), np.float32)
            #print(points)

            cv.line(image, (x_points[0], y_points[0]), (x_points[1], y_points[1]), (0, 255, 0), 2)
            cv.line(image, (x_points[1], y_points[1]), (x_points[2], y_points[2]), (0, 255, 0), 2)
            cv.line(image, (x_points[2], y_points[2]), (x_points[3], y_points[3]), (0, 255, 0), 2)
            cv.line(image, (x_points[3], y_points[3]), (x_points[0], y_points[0]), (0, 255, 0), 2)

            M = cv.getPerspectiveTransform(points[:-1], real_points)

            buoy_pos = np.append(points[4], np.array([1]))
            buoy_pos = np.matmul(M, np.transpose(buoy_pos))
            buoy_pos = buoy_pos / buoy_pos[-1]

            text_str = str(round(buoy_pos[0], 2)) + " " + str(round(buoy_pos[1], 2)) + " "

            transformed_points = []


            cv.line(image, (int(pA[0]), int(pA[1])), (int(pB[0]), int(pB[1])), (0, 255, 0), 2)

            for row in angle_vect:
                point = np.matmul(M, np.transpose(row))
                point = point / point[-1]
                print(point)
                transformed_points.append(point)


            #print(transformed_points)

            angle = np.arctan((transformed_points[1][1] - transformed_points[0][1]) / (
                        transformed_points[1][0] - transformed_points[0][0]))

            angle = angle * (180 / np.pi)

            angle = round(angle, 3)

            text_str += str(angle)

            row = np.array([datetime.now().strftime('%M:%S.%f')[:-4],float(buoy_pos[0]),float(buoy_pos[1]),angle])
            #print(row)
            points_dataset = np.vstack((points_dataset, row))

            writer.writerow(row)

            cv.putText(image, text_str, (x_points[4], y_points[4]), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

    ret, image = cap.read()
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dst, (w, h), 1, (w, h))
    undistored = cv.undistort(image, mtx, dst, None, newcameramtx)
    cv.imwrite("data/UNIDSTORTED.jpg",undistored)
    print("image saved")

    while True:


       # data = input("Enter mm to move, to move forward preface with 'f' for packward preface with 'b': ")
        #s.send(data.encode())
        #recieved_data = s.recv(1024).decode()
        #if not recieved_data:
        #    raise Exception("Did not get a response from tube")
        #else:
        #    pass

        file_name = datetime.now().strftime("%d.%m-%H:%M:%S")
        full_file_name = "data/run@" + file_name + ".csv"

        csv_file = open(full_file_name, 'w')  # Create csv file to store the data collected
        writer = csv.writer(csv_file)

        while True:

            # Capture frame-by-frame
            ret, image = cap.read()  # Ret is a bool that is true if the frame is read correclty
            # if frame is read correctly ret is True
            camerawriter.write(image)
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            # Our operations on the frame come here
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            undistored = cv.undistort(gray, mtx, dst, None, newcameramtx)
            options = apriltag.DetectorOptions(families="tag16h5")
            detector = apriltag.Detector(options)
            results = detector.detect(undistored)
            # Display the resulting frame
            points_dataset = bounding_box(results, image, real_points, points_dataset)
            print(points_dataset[-1])
            if cv.waitKey(1) == ord('q'):
                break
        # When everything done, release the capture


        points_dataset = np.c_[points_dataset]

        print("run saved in: ", file_name)
        save_data = input("Do you wish to save file and plot data? Y/N: ")
        if save_data == "N" or save_data == "n":
            system("rm " + full_file_name)
            quit()
        else:
            print(points_dataset)
            print(points_dataset[:,1][1:])
            print(points_dataset[:,2][1:])
            y_values = []
            x_values = []
            for value in points_dataset[:,2][1:]:
                y_values.append(float(value))
            for value in points_dataset[:,1][1:]:
                x_values.append(float(value))

            plt.scatter(x=x_values, y=y_values)
            plt.axis([real_points[0][0],real_points[2][0],real_points[2][1],real_points[0][1]])
            plt.xlabel("X coordinates (mm)")
            plt.ylabel("Y coordinates (mm)")
            plt.title("Trajectory at: " + file_name)
            plt.show()

        start_again = input("Do you wish to run program again? Y/N: ")
        if start_again == "N" or start_again == "n":
            cap.release()
            cv.destroyAllWindows()
            camerawriter.release()
            exit()
        else:
            pass

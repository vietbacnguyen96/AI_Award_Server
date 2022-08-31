import numpy as np
import struct as st
import torch
import torch.nn as nn
import math
from numpy import dot, sqrt
from numpy.linalg import norm
import cv2

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def norm_angle(angle):
    norm_angle = sigmoid(10 * (abs(angle) / 45.0 - 1))
    # norm_angle = math.cos((angle * math.pi) / 180.0)
    return norm_angle

class Branch(nn.Module):
    def __init__(self, feat_dim):
        super(Branch, self).__init__()
        self.fc1 = nn.Linear(feat_dim, feat_dim)
        self.fc2 = nn.Linear(feat_dim, feat_dim)
        # self.fc3 = nn.Linear(feat_dim, feat_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, yaw):
        x = self.fc1(input)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        # x = self.fc3(x)
        # x = self.relu(x)

        # print('yaw.shape: ' + str(yaw.shape))
        # print('x.shape: ' + str(x.shape))
        # print('Input:      ' + str(input[0][0:5]))
        # print('x:          ' + str(x[0][0:5]))
        yaw = yaw.view(yaw.size(0),1)
        yaw = yaw.expand_as(x)
        # print('yaw:        ' + str(yaw[0][0:5]))
        
        feature = yaw * x + input
        # print('feature:    ' + str(feature[0][0:5]) + '\n')
        return feature
def find_yaw(pts):
    le2n = pts[2] - pts[0]
    re2n = pts[1] - pts[2]
    return le2n - re2n

def draw_landmarks(frame, bb, points):
    # draw rectangle and landmarks on face
    cv2.rectangle(frame,(int(bb[0]),int(bb[1])),(int(bb[2]),int(bb[3])),orange,2)
    cv2.circle(frame, (int(points[0]), int(points[5])), 2, (255,0,0), 2)# eye
    cv2.circle(frame, (int(points[1]), int(points[6])), 2, (255,0,0), 2)
    cv2.circle(frame, (int(points[2]), int(points[7])), 2, (255,0,0), 2)# nose
    cv2.circle(frame, (int(points[3]), int(points[8])), 2, (255,0,0), 2)# mouth
    cv2.circle(frame, (int(points[4]), int(points[9])), 2, (255,0,0), 2)
    
    w = int(bb[2])-int(bb[0])# width
    h = int(bb[3])-int(bb[1])# height
    w2h_ratio = w/h# ratio
    eye2box_ratio = (int(points[0])-bb[0]) / (bb[2]-points[1])
    
    cv2.putText(frame, "Width (pixels): {}".format(w), (10,30), font, font_size, red, 1)
    cv2.putText(frame, "Height (pixels): {}".format(h), (10,40), font, font_size, red, 1)
    
    if w2h_ratio < 0.7 or w2h_ratio > 0.9:
        #cv2.putText(frame, "width/height: {0:.2f}".format(w2h_ratio), (10,40), font, font_size, blue, 1)
        cv2.putText(frame, "Narrow Face", (10,60), font, font_size, red, 1)
    if eye2box_ratio > 1.5 or eye2box_ratio < 0.88:
        #cv2.putText(frame, "leye2lbox/reye2rbox: {0:.2f}".format((points[0]-bb[0]) / (bb[2]-points[1])), (10,70), font, font_size, red, 1)
        cv2.putText(frame, "Acentric Face", (10,70), font, font_size, red, 1)

def one_face(frame, bbs, pointss):
    # process only one face (center ?)
    offsets = [(bbs[:,0]+bbs[:,2])/2-frame.shape[1]/2,
               (bbs[:,1]+bbs[:,3])/2-frame.shape[0]/2]
    offset_dist = np.sum(np.abs(offsets),0)
    index = np.argmin(offset_dist)
    bb = bbs[index]
    points = pointss[:,index]
    return bb, points

def face_orientation(frame, landmarks):
    # print(landmarks)
    size = frame.shape #(height, width, color_channel)
    # index_6_point = [36, 45, 30, 8, 48, 54]
    image_points = np.array([
                            (landmarks[30][0], landmarks[30][1]),     # Nose tip
                            (landmarks[8][0], landmarks[8][1]),   # Chin
                            (landmarks[36][0], landmarks[36][1]),     # Left eye left corner
                            (landmarks[45][0], landmarks[45][1]),     # Right eye right corne
                            (landmarks[48][0], landmarks[48][1]),     # Left Mouth corner
                            (landmarks[54][0], landmarks[54][1])      # Right mouth corner
                        ], dtype="double")
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])

    # Camera internals
 
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)#, flags=cv2.CV_ITERATIVE)

    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (image_points[0][0], image_points[0][1]), image_points



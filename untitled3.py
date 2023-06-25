import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

# initializing mediapipe pose class
mp_pose = mp.solutions.pose

# setting up the pose function
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=0)

# initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils



def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    args:
        image: the input image with a prominent person whose pose landmarks needs to be detected.
        pose: the pose setup function required to perform the pose detection
        display: a boolean value that is if set to true the function displays the original input image, the resultant
                 and the pose landmarks in 3d plot and returns nothing
    
    returns:
        output_image: the input image with the detected pose landmarks drawn.
        landmarks: a list of detected landmarks converted into their original scale.
    '''
    
    #create a copy of the imput image
    output_image = image.copy()
    
    #convert the image from BGR into RBG format
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #perform the pose detection
    results = pose.process(imageRGB)
    
    #retrieve the height and width of the input image
    height, width, _ = image.shape
    
    #initialize a list to sstore the detected landmarks
    landmarks = []
    
    #check if any landmarks are detected
    if results.pose_landmarks:
        
        #draw pose landmarks on the output image
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,connections=mp_pose.POSE_CONNECTIONS)
        
        #itrate over the detected landmarks
        for landmark in results.pose_landmarks.landmark:
            
            #append the landmark into the list
            landmarks.append((int(landmark.x * width), int(landmark.y *height),(landmark.z * width)))
            
    
    #check if the origianl input image and the resultant image are specified to be displayed
    if display:
        
        #display the origianl input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        #also plot the pose landmarks in 3D
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
        
    #otherwise
    else:
        
        #return the output image and the found landmarks.
        return output_image, landmarks




#sample_img = cv2.imread('/Users/Hamed/Pictures/Screenshots\Screenshot (55).png')
#
#plt.figure(figsize = [10,10])
#
#plt.title("Sample Image"); plt.axis('off');plt.imshow(sample_img[:,:,::-1]);plt.show()



        

def calculateAngle(landmark1, landmark2, landmark3):
    '''
    This function calculates angle between three different landmarks.
    Args:
        landmark1: the first landmark containing the x,y and z coordinates
        landmark1: the second landmark containing the x,y and z coordinates
        landmark1: the third landmark containing the x,y and z coordinates
    Returns:
        angle: the calculated angle between the three landmarks
    '''
    
    #get the requireed landmarks coordinates.
    x1,y1, _ = landmark1
    x2,y2, _ = landmark2
    x3,y3, _ = landmark3
    
    #calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    #check if the angle is less than zero
    if angle < 0:
        
        #add 360 to the found angle
        angle += 360
        
    return angle



def classifyPose(landmarks, output_image, display = False):
    '''
    This function classifies yoga poses depending upon the angles of various body joints.
    Args:
        landmarks: a list of detected landmarks of the person whose pose needs to be classified.
        output_image: an image of the person with the detected pose landmarks drawn
        display: a boolean value that is if set to true th function displays the resultant image with the pose label written on it and returns nothing
    Returns:
        output_image: the image with the detected pose landmarks drawn and pose label written.
        label: the classified pose label of the person in the output_image.
    '''
    
    #initialize the lable of the pose. it is not known at this stage.
    label = "Unknown Pose"
    
    #specify the color (Red) with which the label will be written on the image.
    color = (0,0,255)
    
    #calculate the required angles
    #------------------------------------------------------------------------------------------------------------------
    
    #get the angle between the left shoulder, elbow and wrist points
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], 
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    
    #get the angle between the right shoulder, elbow and wrist points
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])
    
    #get the angle between the left elbow, shoulder and hip points
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], 
                                      landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
                                      landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    
    #get the angle between the right elbow, shoulder and hip points
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value])
    
    #get the angle between the left hip, knee and ankle points
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value], 
                                      landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value], 
                                      landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    
    #get the angle between the right hip, knee and ankle points
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value], 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value], 
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    
    #------------------------------------------------------------------------------------------------------------------
    
    
    #check if it is the warrior II pose or the T pose.
    #as for both of them, both arms should be straight and shoulder should be at the specific angle
    
    #------------------------------------------------------------------------------------------------------------------
    
    
    #check if both arms are straight
    if left_elbow_angle > 165 and left_elbow_angle < 195  and right_elbow_angle > 165 and right_elbow_angle < 195:
        
        #check if shoulders are at the required angle
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 250 and right_shoulder_angle < 290:
            
            
    #check if it is the warrior II pose
    #------------------------------------------------------------------------------------------------------------------
            #check if one leg is straight
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                
                #check if the other leg is bended at the required angle
                if left_knee_angle > 90 and left_knee_angle < 120 or right_knee_angle > 90 and right_knee_angle < 120:
                    
                    #specify the label of the pose that is Warrior II pose.
                    label = 'Warrior II Pose'
                    
    
    #------------------------------------------------------------------------------------------------------------------
    
    #check if it is T pose
    
    #------------------------------------------------------------------------------------------------------------------
    
    
            #check if both legs are straight
            if left_knee_angle > 165 and left_knee_angle < 195 and right_knee_angle > 165 and right_knee_angle < 195:
                
                #specify the label as T pose
                label = 'T Pose'
                
    
    #------------------------------------------------------------------------------------------------------------------
    #check if it is the tree pose
    if left_elbow_angle > 40 and left_elbow_angle < 85  and right_elbow_angle > 280 and right_elbow_angle < 330:
        
        #check if shoulders are at the required angle
        if left_shoulder_angle > 80 and left_shoulder_angle < 110 and right_shoulder_angle > 250 and right_shoulder_angle < 290:
        
            if left_knee_angle > 165 and left_knee_angle < 195 or right_knee_angle > 165 and right_knee_angle < 195:
                
                #check if the other leg is bended at the required angle.
                if left_knee_angle > 315 and left_knee_angle < 335 or right_knee_angle > 315 and right_knee_angle < 335:
                    
                    #specify the label of the pose that is tree pose
                    label = 'Tree Pose'
            
    
    #------------------------------------------------------------------------------------------------------------------
    #check if the pose is classified successfully
    if label != 'Unknown Pose':
        
        #update the color (to green) with which the label will be written on the image
        color = (0,255,0)
    
    #write the label on the output image
    cv2.putText(output_image,label,(10,30), cv2.FONT_HERSHEY_PLAIN,2,color,2)
    
    #check if the resultant image is specified to be displayed.
    if display:
        
        #display the resultant image
        plt.figure(figsize=[10,10])
        plt.imshow(output_image[:,:,::-1]);plt.title("Output Image"); plt.axis('off');
    
    else:
        
        #return the output image and the classified label
        return output_image,label
    

# setup pose function for video
pose_video = mp_pose.Pose(static_image_mode=False,min_detection_confidence=0.5,model_complexity=0)

# initialize the VideoCapture object to read from the webcam
camera_video = cv2.VideoCapture(0)

cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

#initialize the VideoCapture Object to read from a video stored in the disk
#video = cv2.VideoCapture('path')

#set video camera size
camera_video.set(3,1280)
camera_video.set(4,960)


#iterate until the video is accessed successfully
while camera_video.isOpened():
    
    #read a frame
    ok, frame = camera_video.read()
    
    #check if frame is not read properly
    if not ok:
        
        #continue the loop
        continue
    
    # flip the frame horizontally for natural (selfie view) visualiztion
    frame = cv2.flip(frame,1)
    
    #get the width and height of the frame
    frame_height, frame_width, *others = frame.shape
    
    #resize the frame while keeping the aspect ratio
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    
    #perform pose landmark detection
    frame, landmarks = detectPose(frame, pose_video,display = False)
    
    
    #check if the difference between the previous and this frame time > 0 to avoid division by zero
    if landmarks:
        
        #calculate the number of frames per second
        frame, _ = classifyPose(landmarks, frame, display=False)
        
        
    
    
    #display the frame
    cv2.imshow('Pose Classification', frame)
    
    #wait until a key is pressed
    #retrieve the ASCII code of the key pressed
    k=cv2.waitKey(1) & 0xFF
    
    #check if 'ESC' is pressed
    if(k == 27):
        
        #break the loop
        break
    
#release the VideoCapture object.
camera_video.release()   

#close the windows
cv2.destroyAllWindows() 
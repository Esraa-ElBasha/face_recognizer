'''
Face recognition :
this code reads the database at the begining where the recognizer is trained on the existing database
,we assumed that every person has two picture fron view but increasig  the data base for every perosn from different views increases the success rate,
then it compares certain image with the database to know who it is since we donot have a database yet we tested this code on yalefaces database
you have to change the path of the existing database,you also have to enter the location of the image you want it to compare it to at the prompt.

We also took some photo and tried the code it word with success 9/10.


creators :
1-Esraa El-Basha
2-Marwan Ibrahim

'''


#!/usr/bin/python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image



# For face detection we will use the Haar Cascade provided by OpenCV.
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

# For face recognition we will the the LBPH Face Recognizer 
recognizer = cv2.createLBPHFaceRecognizer()



########################################################################################################################
#               functions
########################################################################################################################


'''
get_images_and_labels fucntion is responsible for creating the training set for the trainer which is created based on the database it takes path which
is the path of the database
'''
def get_images_and_labels(path):
    # Append all the absolute image paths in a list image_paths
    # We will not read the image with the .sad extension in the training set
    # Rather, we will use them to test our accuracy of the training
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    # images will contains face images
    images = []
    # labels will contains the label that is assigned to the image
    labels = []
    for image_path in image_paths:
        # Read the image and convert to grayscale
        image_pil = Image.open(image_path).convert('L')
        # Convert the image format into numpy array
        image = np.array(image_pil, 'uint8')
        # Get the label of the image
        nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
        # Detect the face in the image
        faces = faceCascade.detectMultiScale(image)
        # If face is detected, append the face to images and the label to labels
        for (x, y, w, h) in faces:
            images.append(image[y: y + h, x: x + w])
            labels.append(nbr)
            cv2.imshow("Adding faces to traning set...", image[y: y + h, x: x + w])
            cv2.waitKey(50)
    # return the images list and labels list
    return images, labels

                                              #############################################################################
'''
image_prediction takes the image_path  parameter which is the path of the image to be recognized 
 '''


def image_prediction(image_path):                  
    counter_above=0
    counter_correct=0
    
    found_flag=0
    predict_image_pil = Image.open(image_path).convert('L')
    predict_image = np.array(predict_image_pil, 'uint8')
    faces = faceCascade.detectMultiScale(predict_image)

  # print(faces)

    for (x, y, w, h) in faces:
            nbr_predicted, conf = recognizer.predict(predict_image[y: y + h, x: x + w])
            nbr_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
           # print(conf)
            
            
            if nbr_actual == nbr_predicted:
                print "{} is Correctly Recognized with confidence {}".format(nbr_actual, conf)
                cv2.imshow("Recognizing Face", predict_image[y: y + h, x: x + w])
                cv2.waitKey(1000)
                found_flag=1;
                counter_correct=counter_correct+1
                
                if conf >= 50:
                    counter_above=counter_above+1
                break
            else:
                print "{} is Incorrect Recognized as {}".format(nbr_actual, nbr_predicted)
        
    if found_flag == 0:
        print('new image')
        
    

#######################################################################################################################################################################

# Path to the Yale Dataset
path = './yalefaces'      ##this is the database path if you have your own database you should change that.

# Call the get_images_and_labels function and get the face images and the 
# corresponding labels

images, labels = get_images_and_labels(path)
cv2.destroyAllWindows()

# Perform the tranining
recognizer.train(images, np.array(labels))

image_path1 = input("Please enter the image path (if you download tha zip file you can try ""./predict\subject02.sad"" between double qoutes")
image_prediction(image_path1)




# Append the images with the extension .sad into image_paths
##image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.sad')]
#image_path = "./predict\subject88.sad"


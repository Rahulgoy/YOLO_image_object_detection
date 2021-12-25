import cv2
import numpy as np

# # paths to the YOLO weights and model configuration
weightsPath = './yolov3.weights'
configPath = './yolov3.cfg'

# load our YOLO object detector trained on COCO dataset (80 classes)
yolo = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


# load the COCO class labels our YOLO model was trained on
classes = []
with open('./coco.names','r') as f:
    classes = f.read().splitlines()

# Other method of reading file in one line
#LABELS = open(labelsPath).read().strip().split("\n")






img = cv2.imread(".images/3.jpg")    #Read the image
# img= cv2.resize(img,(1024,800))
(height, width) = img.shape[:2]

#The arguments passed are:
# img - The image on which operations are to be done
# 1/255 - normalise the image array values so that values lies between 0 and 1
# (320,320) - resizing the image as for a larger image there will be large amount of computations(greater no. of grids will be made)
# swapRB - as cv2 reads image as BGR so to swap the colors back to original
# crop - Don't want to crop the image while resizing
blob = cv2.dnn.blobFromImage(img, 1/255, (320,320), (0,0,0),swapRB=True,crop=False)

# print(blob.shape)
# (1,3,320,320)  no. of images, chaneels, rows, columns


#Set the input data for yolo i.e the resized image stored in blob
yolo.setInput(blob)

ouput_layer_names = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(ouput_layer_names)


boxes=[]
confidences=[]
class_ids=[]

for output in layeroutput:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
    
        #To prevent multiple bounding boxes
        if confidence>0.7:
            # print(detection[:5])
            # print(class_id)
            # print(confidence)
            center_x = int(detection[0]*width)
            # print(center_x)
            center_y = int(detection[1]*height)
            # print(center_y)
            w = int(detection[2]*width)
            h = int(detection[3]*height)

            x = int(center_x - w/2)
            y= int(center_y - h/2)


            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

print(len(boxes))

indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
font = cv2.FONT_HERSHEY_PLAIN
# initialize a list of colors to represent each possible class label
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i in indexes.flatten():
    x,y,w,h = boxes[i]

    label = str(classes[class_ids[i]])
    confid = str(round(confidences[i],2))
    color = colors[i]

    cv2.rectangle(img, (x,y),(x+w,y+h),color, 2)
    # cv2.putText(img, label+" "+confid, (x,y+20),font, 1, (255,255,255),1)

    text = "{}: {:.4f}".format(label, confidences[i])
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
	


# cv2.imwrite("detect1.jpg",img)
cv2.imshow("Image",img)
cv2.waitKey(0)
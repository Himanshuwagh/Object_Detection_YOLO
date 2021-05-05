import numpy as np
import cv2

# Load YOLO
net=cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
classes=[]
with open('coco.names','r') as f:
    classes=[line.strip() for line in f.readlines()]

layer_names=net.getLayerNames()
output_names=[layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
# print(output_names)

# loading images
img=cv2.imread('test_img.jpg')
# img=cv2.resize(img,None,fx=0.2,fy=0.2)
height,width,channel=img.shape

blob=cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=True)

# for b in blob:
#     for n,img_blob in enumerate(b):
#         cv2.imshow(str(n),img_blob)

net.setInput(blob)
outs=net.forward(output_names)

# showing information on the screen

confidences=[]
class_ids=[]
boxes=[]

for out in outs:
    for detection in out:
        scores=detection[5:]
        class_id=np.argmax(scores)
        confidence=scores[class_id]

        if confidence > 0.2:
            # object detected
            center_x=int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # rectangle co-ordinates
            x=int(center_x-w/2)
            y=int(center_y-h/2)

            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x,y,w,h])



number_objects_detected=len(boxes)
indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)

font=cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(boxes)):
    if i in indexes:

        x,y,w,h=boxes[i]
        label=str(classes[class_ids[i]])

        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img,label,(x,y-30),font,1,(0,0,0),2)


cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
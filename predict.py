
from tensorflow import keras
import cv2
import numpy as np


class ASL:
    
    def __init__(self, path_to_model):
        self.model = keras.models.load_model(path_to_model)

    #this function can be used to predict the sign through a given image
    def predict_file(self,imagepath):

        image = cv2.resize(cv2.imread(imagepath), (64, 64))
        img = np.array([image])
        img = img.astype('float32') / 255.0
        ans = self.model.predict(img, batch_size = 64, verbose = 0)[0]
        return np.argmax(ans)

    def predict_image(self, image):
        image = np.array([cv2.resize(frame, (64, 64))])
        image = image.astype('float32') / 255.0
        ans = self.model.predict(image, batch_size = 64, verbose = 0)[0]
        return [np.argmax(ans), image[0]]
        
asl = ASL("aslmodel")
cap = cv2.VideoCapture(0)
#cap.set(3, 1280)
#cap.set(4, 720)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    [result , image] = asl.predict_image(frame)
    if result== 0:
        print("A")
    elif result ==1:
        print('B')
    elif result ==2:
        print('C')
    elif result ==3:
        print('D')
    elif result ==4:
        print('E')
    elif result ==5:
        print('F')
    elif result ==6:
        print('G')
    elif result ==7:
        print('H')
    elif result ==8:
        print('I')
    elif result ==9:
        print('J')
    elif result ==10:
        print('K')
    elif result ==11:
        print('L')
    elif result ==12:
        print('M')
    elif result ==13:
        print('N')
    elif result ==14:
        print('O')
    elif result ==15:
        print('P')
    elif result ==16:
        print('Q')
    elif result ==17:
        print('R')
    elif result ==18:
        print('S')
    elif result ==19:
        print('T')
    elif result ==20:
        print('U')
    elif result ==21:
        print('V')
    elif result ==22:
        print('W')
    elif result ==23:
        print('X')
    elif result ==24:
        print('Y')
    elif result ==25:
        print('Z')
    else:
        print('none')
   
    # Display the resulting frame
    #cv2.imshow("123",image)
    frame = rescale_frame(frame, percent=150)
    cv2.imshow('frame75', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()





import tensorflow as tf 
import matplotlib as plt
import numpy
import cv2
import time

class student:
    name = None #Namn på elev
    rightGazeDirection = None #Hur mycket har eleven kollat åt rätt håll
    wrongGazeDirection = None #Hur mucket har eleven kollat åt fel håll
    avgGazeDirection = None
    avgEyelid = None
    timeGraph = None #Base64 för grafen till hemsidan

Students = (student) #skapar en elev

RED = (255, 0, 0) #röd färg
GREEN = (0, 255, 0) #grön färg
BLUE = (0, 0, 255) #blå färg

startTime = None #start klockslag
endTime = None #slut klockslag

lastPrediction = None #Förra blickriktningen

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') #OpenCV model för att hitta ansikten


face_classifier = tf.keras.models.load_model('models\GTG_tracker_test4') #ladda modell för klasifiering av ansiktet

class_names = ['left', 'forward','right'] #namn på de olika klasserna för klassifiering

def get_extended_image(img, x, y, w, h, k=0.1): #stulen funktion, se föklaring nedan
    '''
    Function, that return cropped image from 'img'
    If k=0 returns image, cropped from (x, y) (top left) to (x+w, y+h) (bottom right)
    If k!=0 returns image, cropped from (x-k*w, y-k*h) to (x+k*w, y+(1+k)*h)
    After getting the desired image resize it to 250x250.
    And converts to tensor with shape (1, 250, 250, 3)
    Parameters:
        img (array-like, 2D): The original image
        x (int): x coordinate of the upper-left corner
        y (int): y coordinate of the upper-left corner
        w (int): Width of the desired image
        h (int): Height of the desired image
        k (float): The coefficient of expansion of the image

    Returns:
        image (tensor with shape (1, 250, 250, 3))
    '''

    # The next code block checks that coordinates will be non-negative
    # (in case if desired image is located in top left corner)
    if x - k*w > 0:
        start_x = int(x - k*w)
    else:
        start_x = x
    if y - k*h > 0:
        start_y = int(y - k*h)
    else:
        start_y = y

    end_x = int(x + (1 + k)*w)
    end_y = int(y + (1 + k)*h)

    face_image = img[start_y:end_y,
                     start_x:end_x]
    face_image = tf.image.resize(face_image, [250, 250])
    # shape from (250, 250, 3) to (1, 250, 250, 3)
    face_image = numpy.expand_dims(face_image, axis=0)
    return face_image

video_capture = cv2.VideoCapture(0)  # webkamera

if not video_capture.isOpened():
    print("Unable to access the camera")
else:
    print("Access to the camera was successfully obtained")

print("Streaming started - to quit press ESC")
while True:

    
    ret, frame = video_capture.read() #sparar bilden
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #konverterar till gråskala

    faces = face_cascade.detectMultiScale( #detekterar objekt av olika storlek
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        face_image = get_extended_image(frame, x, y, w, h, 0.5) #plockar ut ansiktet ur bilden
        
        result = face_classifier.predict(face_image) #Klassifierar ansiktet
        prediction = class_names[numpy.array(
            result[0]).argmax(axis=0)]  

        if prediction == 'left': #om eleven tittar åt vänster
            if lastPrediction == 'right' or lastPrediction == 'not': #om eleven tittade åt fel håll förut
                endTime = time.time() #tid för bytet
                Students[0].wrongGazeDirection += (endTime - startTime) #räknar upp hur länge man tittade fel
                startTime = time.time() #startar räkningen igen
                lastPrediction = 'left' #ändrar värdet för förra riktningen
            color = GREEN
        elif prediction == 'forward': #om eleven tittar framåt
            if lastPrediction == 'right' or lastPrediction == 'not': #om eleven tittade åt fel håll förut
                endTime = time.time() #tid för bytet
                Students[0].wrongGazeDirection += (endTime - startTime) #räknar upp hur länge man tittade fel
                startTime = time.time() #startar räkningen igen
                lastPrediction = 'forward' #ändrar värdet för förra riktningen
            color = RED
        elif prediction == 'right': #likt förra men skiftet sker från rätt till fel
            if lastPrediction == 'left' or lastPrediction == 'forward':
                endTime = time.time()
                Students[0].rightGazeDirection += (endTime - startTime)
                startTime = time.time()
            lastPrediction = 'right'
            color = BLUE
        else: 
            if lastPrediction == 'left' or lastPrediction == 'forward': #som förra if satsen
                endTime = time.time()
                Students[0].rightGazeDirection += (endTime - startTime)
                startTime = time.time()
            lastPrediction = 'not'
        
        cv2.rectangle(frame, #ritar rektagel runt ansiktet
                      (x, y),  # start punkt
                      (x+w, y+h),  # slut punkt
                      color,
                      2)  # tjocklek
        cv2.putText(frame, #skriver ut vart man tittar ovanför rektangeln
                    "{:6}".format(prediction),
                    (x, y),
                    cv2.FONT_HERSHEY_PLAIN,  # typsnitt
                    2,  #storlek
                    color,
                    2)  # tjocklek

    cv2.imshow("Face detector - to quit press ESC", frame) #visar bilden

    key = cv2.waitKey(1) #stäng ner om ma trycker på esc
    if key % 256 == 27:  
        break

video_capture.release() #färdig :)
cv2.destroyAllWindows()
print("Streaming ended")
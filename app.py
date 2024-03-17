from flask import Flask, flash ,render_template, Response, session, request, send_file, sessions
import cv2
import numpy as np
import mediapipe as mp
import time
from ultralytics import YOLO
import math
import dlib
import os
from flask_bootstrap import Bootstrap
from flask_mysqldb import MySQL
from PIL import Image
import json

# Load the pre-trained face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")

cheat = 0
lips = 0
direction=''
cellphone=0
identity=''
liveness = ''

# Capturing User Image details
name=''
id=''
capture_enabled = False


# Initialize Flask app
app = Flask(__name__)

app.secret_key = 'your_very_secret_key'

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:@localhost/proctoring'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # recommended for performance

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy(app)

# # DataregNumnerbase Connection
# app.config['MYSQL_HOST'] = 'localhost'
# app.config['MYSQL_USER'] = 'root'
# app.config['MYSQL_PASSWORD'] = ''
# app.config['MYSQL_DB'] = 'proctoring'

# mysql = MySQL(app)


Bootstrap(app)

# Class Loading------------------------------------------------------------------------------------------------------------------------------

class Users(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))
    regNumber = db.Column(db.String(255))
    

    def __init__(self, name, regNumber):  # Accept both arguments
        self.name = name
        self.regNumber = regNumber

    def __repr__(self):
        return f"<Answer {self.name}>"

class Questions(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(500))
    answerId = db.Column(db.Integer, db.ForeignKey('answers.id'))
    courseId = db.Column(db.Integer, db.ForeignKey('course.id'))
    lecturerId = db.Column(db.Integer, db.ForeignKey('lecturers.id'))
    

    def __init__(self,question, courseId, lecturerId):  # Accept both arguments
        self.question = question
        self.courseId = courseId
        self.lecturerId = lecturerId

    def __init2__(self, answerId):  # Accept both arguments
        self.answerId = answerId

    def __repr__(self):
        return f"<Question {self.question}>"

class Answers(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    answer = db.Column(db.String(255))
    questionId = db.Column(db.Integer, db.ForeignKey('questions.id', ondelete='CASCADE'))
    

    def __init__(self, answer, questionId):  # Accept both arguments
        self.answer = answer
        self.questionId = questionId

    def __repr__(self):
        return f"<Answer {self.answer}>"

class Course(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    courseTitle = db.Column(db.String(255))
    lecturerId = db.Column(db.Integer, db.ForeignKey('lecturers.id', ondelete="SET NULL"))
    examId = db.Column(db.Integer, db.ForeignKey('exam.id', ondelete="CASCADE"))

    def __init__(self, courseTitle, lecturerId):  # Accept both arguments
        self.courseTitle = courseTitle
        self.lecturerId = lecturerId

    def __repr__(self):
        return f"<Answer {self.courseTitle}>"

class Lecturers(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(255))

    def __init__(self, name):  # Accept both arguments
        self.name = name

    def __repr__(self):
        return f"<Answer {self.name}>"

class Exam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(255))
    courseId = db.Column(db.Integer, db.ForeignKey('course.id'))

    def __init__(self, code,courseId):  # Accept both arguments
        self.code = code
        self.courseId = courseId

    def __repr__(self):
        return f"<Answer {self.answer_text}>"

with app.app_context():  # Ensure we're in the application context
      db.create_all()

# Class Loading------------------------------------------------------------------------------------------------------------------------------

# Load the cascade classifier for face detection (outside of routes for efficiency)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Specify the path to save the images
save_path = os.path.join(app.root_path, 'static','images')  # Ensure path is relative to app root
os.makedirs(save_path, exist_ok=True)


# HeadPose Estimation
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Face Recognition
# Fetching Users from DB
def get_user_data():
    with app.app_context():
        student_data = Users.query.all()

        # Check if there are no records in the database
        if not student_data:
            # If no records, initialize with default values
            default_user = Users(name="Unknown", regNumber="0000")
            student_data = [default_user]
    return student_data


def video_detection():
    global cheat 
    global lips 
    global direction
    global cellphone 
    global identity
    global liveness

    confidence = 0.5
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Load both YOLO models
    model_object = YOLO("./models/yolov8n.pt")
    model_liveness = YOLO("./models/best_20.pt")  # Path to your liveness detection model

    # Face Recognition
    recognizer = cv2.face.LBPHFaceRecognizer_create()  # cv2.createLBPHFaceRecognizer()
    recognizer.read("./models/TrainingImageLabel/Trainner.yml")
    harcascadePath = "./models/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    classNames_object = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]
    classNames_liveness = ["real", "fake"]

    
    while True:
        success, img = cap.read()

# Head Pose Estimation Opening
        # Flip the image for a selfie-view display
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

        
        # Create dictionary to store student data
        with app.app_context():
            students = {}
            for row in get_user_data():
                students[row.regNumber] = {"Id": row.id, "Name": row.name}
                pass


# Face Recognition
        faces = faceCascade.detectMultiScale(img, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (225, 0, 0), 2)
            gray = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            serial, confa = recognizer.predict(gray)
#             # secondCamera(prev_frame_time,new_frame_time,gray)
            if (confa > 50):
                # serial = reg_number  # Assuming reg_number is extracted from the face
                if serial in students:
                    bb = students[serial]["Name"]
                    ID = students[serial]["Id"]
                else:
                    Id = 'Unknown'
                    bb = str(Id)
            else:
                bb = "Unknown"  # Assign default value if registration not found

            cv2.putText(img, str(bb), (x, y + h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img, str(bb), (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
       
            identity = bb

# Mouth Detection

        # Detect faces in the frame
        faces = detector(img)
        
        for face in faces:
            landmarks = predictor(img, face)
            
            # Extract mouth landmarks (assuming 68-point facial landmark model)
            mouth_left = landmarks.part(48).x, landmarks.part(48).y
            mouth_right = landmarks.part(54).x, landmarks.part(54).y
            mouth_top = landmarks.part(51).x, landmarks.part(51).y
            mouth_bottom = landmarks.part(57).x, landmarks.part(57).y
            
            # Calculate the distance between top and bottom lip to determine if mouth is open or closed
            lip_distance = mouth_bottom[1] - mouth_top[1]
            print(lip_distance)

            # Display if the mouth is open or closed based on lip distance
            if lip_distance > 21:  # You can adjust this threshold based on your needs
                cv2.putText(img, "Mouth Open", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                lips = "Mouth Open"
            else:
                cv2.putText(img, "Mouth Closed", (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                lips = "Mouth Closed"

    # Mouth Detection

        # To improve performance
        img.flags.writeable = False
        
        # Get the result
        faceResults = face_mesh.process(img)
        
        # To improve performance
        img.flags.writeable = True

        # Convert the color space from RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = img.shape
        face_3d = []
        face_2d = []

        if faceResults.multi_face_landmarks:
            for face_landmarks in faceResults.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
            

                # See where the user's head tilting
                # Assigning cheat values based on face direction
                if y < -10:
                    text = "Looking Left"
                    cheat = 0.4
                elif y > 10:
                    text = "Looking Right"
                    cheat = 0.4
                elif x < -10:
                    text = "Looking Down"
                    cheat = 0.8
                elif x > 10:
                    text = "Looking Up"
                    cheat = 0.5
                else:
                    text = "Forward"
                    cheat = 0.15
                direction = text
                print("Cheat 1:",cheat)

                # Add the text on the image
                cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                cv2.putText(img, str(cheat), (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                
# Head Pose Estimation Closure

        # Perform object detection
        results_object = model_object(img, stream=True)
        for r in results_object:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames_object[cls]
                label = f'{class_name}{conf}'
                cv2.putText(img, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

# Cellphone detected
                if (class_name == "cellphone"):
                    cellphone = 1

       # Perform liveness detection
        num_people = 0
        face_data = {}

        results_liveness = model_liveness(img, stream=True)
        for r in results_liveness:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                # class_name = classNames_liveness[cls]

                col = (0, 255, 0)
                if conf > confidence:
                    if classNames_liveness[cls] == 'real':
                        col = (0, 255, 0)
                    else:
                        col = (0, 0, 255)

                # Red bounding box for "fake" liveness
                # col = (0, 0, 255) if class_name == "fake" else (0, 255, 0)
                
                cv2.rectangle(img, (x1, y1), (x2, y2), col, 3)
                label = f'{classNames_liveness[cls]}{conf}'
                liveness = classNames_liveness[cls]
                cv2.putText(img, label, (x1, y1-2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
               
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
    cap.release()

# Capture Images
@app.route('/generate_frames')
def generate_frames():
    text_input = request.form['text']  # Retrieve text input from form
    idInput = request.form['id']

    # Adding Question to DB
    my_data = Users(text_input,idInput)
    db.session.add(my_data)
    db.session.commit()

    confidence = 0.5
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    count = 0
    while True:
        success, frame = cap.read()    
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            if len(faces) > 0 and count < 100:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    face_img = frame[y:y + h, x:x + w]
                    image_title = text_input+"."+idInput+".{}.jpg".format(count)
                    image_path = os.path.join(save_path, image_title)
                    success = cv2.imwrite(image_path, face_img)
                    if success:
                        count += 1

                    if count >= 100:
                        break

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Capture Images

# Graph----------------------------------------------------------------------------

# ------------------------Views

# Humidity

try:

    from flask import (Blueprint,
                       render_template,
                       redirect, url_for, session)

    from flask import Flask, request, session, send_file
    import json
    from time import time
    from random import random
    from flask import Flask, render_template, make_response

except Exception as e:
    print("Some modules didnt load {}".format(e))

humidity_blueprint = Blueprint('Humidity', __name__)


@humidity_blueprint.route('/humidity', methods=['GET'])
def humidity():
    global cheat
    # Create a PHP array and echo it as JSON

    data = [time() * 1000, cheat ]
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    return response

# Sensor

try:

    from flask import (Blueprint,
                       render_template,
                       redirect, url_for, session)

    from flask import Flask, request, session, send_file
    import random
    import json
    from time import time
    from random import random
    from flask import Flask, render_template, make_response

except Exception as e:
    print("Some modules didnt load {}".format(e))

sensor_blueprint = Blueprint('Sensor', __name__)


@sensor_blueprint.route('/data', methods=['GET'])
def data():
    global cheat
    Temperature = []
    for i in range(1,10):
        # Temperature.append(random())
        Temperature.append(cheat)
    data = {
        "temperature":Temperature
    }
    return data




# Humidity

try:

    from flask import (Blueprint,
                       render_template,
                       redirect, url_for, session)

    from flask import Flask, request, session, send_file
    import random
    import json
    from time import time
    from random import random
    from flask import Flask, render_template, make_response

except Exception as e:
    print("Some modules didnt load {}".format(e))

humidity_blueprint = Blueprint('Humidity', __name__)


@humidity_blueprint.route('/humidity', methods=['GET'])
def humidity():
    global cheat
    # Create a PHP array and echo it as JSON
    data = [time() * 1000, cheat ]
    response = make_response(json.dumps(data))
    response.content_type = 'application/json'
    return response

# Sensor

try:

    from flask import (Blueprint,
                       render_template,
                       redirect, url_for, session)

    from flask import Flask, request, session, send_file
    import random
    import json
    from time import time
    from random import random
    from flask import Flask, render_template, make_response

except Exception as e:
    print("Some modules didnt load {}".format(e))

sensor_blueprint = Blueprint('Sensor', __name__)


@sensor_blueprint.route('/data', methods=['GET'])
def data():
    Temperature = []
    for i in range(1,10):
        Temperature.append(cheat)
    data = {
        "temperature":Temperature
    }
    return data



# Init

try:
    import os
    from flask import Flask
    from flask import (Flask,
                       request,render_template,
                       redirect,
                       url_for,
                       session,
                       send_file)

    # from views import humidity_blueprint, sensor_blueprint

except Exception as e:
    print("Some Modules are Missing {}".format(e))

app.config["SECRET_KEY"] = "mysecretkey"
app.register_blueprint(sensor_blueprint, url_prefix="/Sensor")
app.register_blueprint(humidity_blueprint, url_prefix="/Humidity")

# app.register_blueprint(result_blueprint, url_prefix="/Result")


# App

try:
    from flask import render_template

    from flask import (Blueprint,
                       render_template,
                       redirect, url_for)

    from flask import (Flask,
                       request,
                       redirect,
                       session,
                       send_file)

    from io import BytesIO
    from flask import abort, jsonify
    import io
    from  random import sample

except Exception as e:
    print("Failed to load some Modules ")

# Graph----------------------------------------------------------------------------

# Traing Images for facial Recognition
def TrainImages():
    # check_haarcascadefile()
    # assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = ".\models\haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels(".\static\images")
    # try:
    recognizer.train(faces, np.array(ID))
    # except:
    #     mess._show(title='No Registrations', message='Please Register someone first!!!')
    #     return
    recognizer.save(".\models\TrainingImageLabel\Trainner.yml")
    res = "Profile Saved Successfully"
    # message1.configure(text=res)
    # message.configure(text='Total Registrations till now  : ' + str(ID[0]))
    print(res)

def getImagesAndLabels(path):
    # get the path of all the files in the folder
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    # create empth face list
    faces = []
    # create empty ID list
    Ids = []
    # now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        # loading the image and converting it to gray scale
        pilImage = Image.open(imagePath).convert('L')
        # Now we are converting the PIL image into numpy array
        imageNp = np.array(pilImage, 'uint8')
        # getting the Id from the image
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids
# Traing Images for facial Recognition

# Flask logic-----------------------------------------------------------------------

@app.route('/proctor', methods=['GET', 'POST'])
def index():
    student_data = get_user_data()
    
    data = {
        "cheat": cheat,
        "lips": lips,
        "direction": direction,
        "cellphone": cellphone,
        "identity": identity,
        "liveness": liveness
    }
    return render_template('proctor.html',student_data=student_data,data=data)

@app.route('/video')
def video():
    # Create a dictionary with the variables
    return Response(video_detection(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def home():
    return render_template('home.html')


# -----------------------------------------Capturing Image--------------------------------------
@app.route('/captureDetails', methods=['POST'])
def captureDetails():
    global capture_enabled, name, id
    name = request.form['name']  
    id = request.form['id']
    text3 = request.form['text3']
    if text3 == 'True':
        capture_enabled = True

    my_data = Users(name,id)
    db.session.add(my_data)
    db.session.commit()
    return redirect(url_for('captureImage'))

capture=False
@app.route('/captureImage',methods=['GET', 'POST'] )
def captureImage():
    return render_template('capture_images.html')

# ----------------Render----------------
@app.route('/captureRender')
def captureRender():
    # Create a dictionary with the variables
    return Response(captured(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------Yield----------------
@app.route('/captured')
def captured():
    global name, id, capture_enabled
    count = 0
    max_capture_count = 100
    video_capture = cv2.VideoCapture(0)

    while True:        
        success, frame = video_capture.read()  # read the camera frame
        if not success:
            break
        
        detector = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')
        faces = detector.detectMultiScale(frame, 1.1, 7)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]

            # Capturing Images
            if count < max_capture_count and capture_enabled:
                face_img = frame[y:y+h, x:x+w]
                image_title = name+"."+id+".{}.jpg".format(count)
                image_path = os.path.join(save_path, image_title)
                success = cv2.imwrite(image_path, face_img)
                if success:
                    count += 1

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# -----------------------------------------Capturing Image--------------------------------------


# @app.route('/capture', methods=['POST'])
# def capture():
#     text_input = request.form['text']  # Retrieve text input from form
#     idInput = request.form['id']

#     # Adding Question to DB
#     my_data = Users(text_input,idInput)
#     db.session.add(my_data)
#     db.session.commit()

#     # Create a VideoCapture object to access the webcam
#     video_capture = cv2.VideoCapture(0)

#     count = 0
#     while True:
#         ret, frame = video_capture.read()

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

#         if len(faces) > 0 and count < 100:
#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#                 face_img = frame[y:y + h, x:x + w]
#                 image_title = text_input+"."+idInput+".{}.jpg".format(count)
#                 image_path = os.path.join(save_path, image_title)
#                 success = cv2.imwrite(image_path, face_img)
#                 if success:
#                     count += 1

#         if count >= 100:
#             break

#         cv2.imshow("Take Images", frame) 
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#     video_capture.release()
#     cv2.destroyAllWindows()

#     return redirect(url_for('display_images'))  # Redirect to display images

# Obtaining Names for fetching images
def get_names():
    data = Users.query.all()
    return data

@app.route('/display_images')
def display_images():

    data = get_names()
  
    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join("static","images")
    images_dir = os.path.join(current_dir, static_dir)
 
    image_paths = [("static/images/" + filename) for filename in os.listdir(images_dir)]
    return render_template('display_images.html', image_paths=image_paths, data=data)

@app.route('/display_images_admin')
def display_images_admin():

    data = get_names()
    return render_template('display_images_admin.html', data=data)

@app.route('/get_images', methods=['POST'])
def get_images():
    name = request.form.get("name")
    images = []
    if name:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_dir = os.path.join("static", "images")
        images_dir = os.path.join(current_dir, static_dir)

        for filename in os.listdir(images_dir):
            if filename.startswith(name):
                images.append(filename)
                print("Image Paths - "+str(images))
    print (images)
    return json.dumps({"images": images})

@app.route('/train_images', methods=['POST'])
def train_images():
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join("static","images")
    images_dir = os.path.join(current_dir, static_dir)
 
    image_paths = [("static/images/" + filename) for filename in os.listdir(images_dir)]
    if request.method == 'POST':
        # Call the TrainImages function when the button is clicked
        TrainImages()
        message = "Training complete!"  # Adjust the message as needed
        return render_template('display_images.html', image_paths=image_paths, message=message)
    else:
        return render_template('display_images.html', image_paths=image_paths)

@app.route('/procbase')
def base():
    return render_template('proctorbase.html')

@app.route('/post')
def posst():
    return render_template('test.html')

@app.route("/process_selection", methods=["POST"])
def process_selection():
    selected_value = request.form.get("mySelect")
    # Process the selected value here
    return f"You selected: {selected_value}"




# Quiz Flask Logic-----------------------------------------------------------------------------------------------------------
# Questions---------------------------------------------------------------------------------------------------------------------------------------------    
#query on all our questions and answer data
@app.route('/viewQuestions/<int:course_id>', methods = ['GET', 'POST'])
def viewQuestions(course_id):
    # Questions data
    questions = Questions.query.filter_by(courseId=course_id).all()
    courses = Course.query.filter_by(id=course_id).all()
    lecturers = Lecturers.query.all()
    # Answers data
    answers = {}
    for question in questions:
        answers[question.id] = Answers.query.filter_by(questionId=question.id).all()

    return render_template("manageQuestions.html", questions=questions, answers=answers, courses=courses, lecturers=lecturers)


#insert data to mysql database via html forms
@app.route('/createQuestions', methods = ['POST'])
def insert():
    if request.method == 'POST':
        question = request.form['question']
        course = request.form['courseId']
        lecturer = request.form['lecturerId']
        answer1 = request.form['answer1']
        answer2 = request.form['answer2']
        answer3 = request.form['answer3']
        answer4 = request.form['answer4']
  
        # Adding Question to DB
        my_data = Questions(question, course, lecturer)
        db.session.add(my_data)
        db.session.commit()

        # Adding ANswers to DB
        answers = [answer1, answer2, answer3, answer4]
        for answer_text in answers:
            my_answer = Answers(answer_text, questionId=my_data.id)  # Use my_data.id after commit
            db.session.add(my_answer)
        db.session.commit()

        flash("Question Created Successfully")
        return redirect(url_for('viewQuestions',course_id=course))
  
#update questions
@app.route('/updateQuestions', methods = ['GET', 'POST'])
def update():
    if request.method == 'POST':
        my_data = Questions.query.get(request.form.get('id'))
        my_data2 = Answers.query.filter_by(questionId=request.form.get('id')).all()
        course = request.form['courseId']

        my_data.question = request.form['question']
        for answer in my_data2:
            # Update individual answer based on form data
            answer.answer = request.form.get(f"answer_{answer.id}")  # Access answer-specific form field

        db.session.commit()
        flash("Question Updated Successfully")
    return redirect(url_for('viewQuestions',course_id=course))
  

#Delete questions
@app.route('/deleteQuestions/<id>/', methods = ['GET', 'POST'])
def deleteQuestions(id):
    my_data = Questions.query.get(id)
    if my_data:  # Check if questions exists before deleting
        db.session.delete(my_data)
        db.session.commit()
        flash("Questions Deleted Successfully")
    else:
        flash("Questions not found!")
    return redirect(url_for('viewQuestions', course_id = id))



# Questions---------------------------------------------------------------------------------------------------------------------------------------------    


# Lecturers---------------------------------------------------------------------------------------------------------------------------------------------  
#insert data to mysql database via html forms
@app.route('/viewLecturers')
def viewLecturers():
    # Questions data
    all_data = Lecturers.query.all()

    return render_template("manageLecturers.html", lecturers = all_data)
  

#insert data to mysql database via html forms
@app.route('/createLecturers', methods = ['POST'])
def createLecturers():
    if request.method == 'POST':
        lecturer = request.form['lecturerName']
  
        # Adding Question to DB
        my_data = Lecturers(lecturer)
        db.session.add(my_data)
        db.session.commit()

        flash("Question Created Successfully")
        return redirect(url_for('viewLecturers'))

#update lecturers
@app.route('/updateLecturers', methods = ['GET', 'POST'])
def updateLecturers():
    if request.method == 'POST':
        my_data = Lecturers.query.get(request.form.get('id'))
  
        my_data.name = request.form['name']
        db.session.commit()
        flash("Lecturer Data Updated Successfully")
    return redirect(url_for('viewLecturers'))
  

# Delete lecturers
@app.route('/deleteLecturers/<id>/', methods=['GET', 'POST'])
def deleteLecturers(id):
    my_data = Lecturers.query.get(id)
    if my_data:  # Check if lecturer exists before deleting
        db.session.delete(my_data)
        db.session.commit()
        flash("Lecturer Deleted Successfully")
    else:
        flash("Lecturer not found!")
    return redirect(url_for('viewLecturers'))

# Lecturers---------------------------------------------------------------------------------------------------------------------------------------------  


# Course---------------------------------------------------------------------------------------------------------------------------------------------  
#insert data to mysql database via html forms
@app.route('/viewCourses')
def viewCourses():
    # Courses data
    all_data = Course.query.all()
    lecturers = Lecturers.query.all()

    return render_template("manageCourses.html", courses = all_data, lecturers = lecturers)
  
@app.route('/test')
def test():
    return render_template("test.html")
  

#insert data to mysql database via html forms
@app.route('/createCourse', methods = ['POST'])
def createCourse():
    if request.method == 'POST':
        course = request.form['courseName']
        lecturer = request.form['lecturerId']
        # Adding Course to DB
        my_data = Course(course,lecturer)
        db.session.add(my_data)
        db.session.commit()

        flash("Course Created Successfully")
    return redirect(url_for('viewCourses'))

#update Course
@app.route('/updateCourse', methods = ['GET', 'POST'])
def updateCourse():
    if request.method == 'POST':
        my_data = Course.query.get(request.form.get('id'))
  
        my_data.courseTitle = request.form['courseName']
        my_data.lecturerId = request.form['lecturerId']
        db.session.commit()
        flash("Course Data Updated Successfully")
    return redirect(url_for('viewCourses'))
  

# Delete Course
@app.route('/deleteCourse/<id>/', methods=['GET', 'POST'])
def deleteCourse(id):
    my_data = Course.query.get(id)
    if my_data:  # Check if lecturer exists before deleting
        db.session.delete(my_data)
        db.session.commit()
        flash("Lecturer Deleted Successfully")
    else:
        flash("Lecturer not found!")
    return redirect(url_for('viewCourses'))


# Course---------------------------------------------------------------------------------------------------------------------------------------------  


if __name__ == '__main__':
    app.run(debug=True)























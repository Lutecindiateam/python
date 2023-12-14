from flask import Flask, render_template, request, redirect, url_for, Response
from flask_mysqldb import MySQL
import cv2
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import joblib
import dlib
import shutil
import time
import logging
import tkinter as tk
from tkinter import font as tkFont
from PIL import Image, ImageTk
import yaml
import sqlite3
import datetime

app = Flask(__name__)

with open('db.yaml', 'r') as config_file:
    db = yaml.load(config_file, Loader=yaml.SafeLoader)

app.config['MYSQL_HOST'] = db['mysql_host']
app.config['MYSQL_USER'] = db['mysql_user']
app.config['MYSQL_PASSWORD'] = db['mysql_password']
app.config['MYSQL_DB'] = db['mysql_db']
app.config['MYSQL_PORT'] = 3307


mysql = MySQL(app)


knn = KNeighborsClassifier(n_neighbors=5)

known_faces = {}
total_unique_faces = len(known_faces)

detector = dlib.get_frontal_face_detector()

registered_faces = []
registered_users = {}
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

from attendance_taker import Face_Recognizer
from logout_attendance  import Face_atten

# Create an instance of Face_Recognizer
face_recognizer = Face_Recognizer()
face_recog = Face_atten()


class Face_Register:
    def __init__(self):

        self.current_frame_faces_cnt = 0  #  cnt for counting faces in current frame
        self.existing_faces_cnt = 0  # cnt for counting saved faces
        self.ss_cnt = 0  #  cnt for screen shots

        # Tkinter GUI
        self.win = tk.Tk()
        self.win.title("Face Register")

        # PLease modify window size here if needed
        self.win.geometry("1000x500")

        # GUI left part
        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)
        self.frame_left_camera.pack()

        # GUI right part
        self.frame_right_info = tk.Frame(self.win)
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text=str(self.existing_faces_cnt))
        self.label_fps_info = tk.Label(self.frame_right_info, text="")
        self.input_name = tk.Entry(self.frame_right_info)
        self.input_name_char = ""
        self.label_warning = tk.Label(self.frame_right_info)
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ")
        self.log_all = tk.Label(self.frame_right_info)

        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""
        self.font = cv2.FONT_ITALIC

        # Current frame and face ROI position
        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0

        self.out_of_range_flag = False
        self.face_folder_created_flag = False

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        self.cap = cv2.VideoCapture(0)  # Get video stream from camera


    #  Delete old face folders
    def GUI_clear_data(self):
        #  "/data_faces_from_camera/person_x/"...
        folders_rd = os.listdir(self.path_photos_from_camera)
        for i in range(len(folders_rd)):
            shutil.rmtree(self.path_photos_from_camera + folders_rd[i])
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")
        self.label_cnt_face_in_database['text'] = "0"
        self.existing_faces_cnt = 0
        self.log_all["text"] = "Face images and `features_all.csv` removed!"

    def GUI_get_input_name(self):
        self.input_name_char = self.input_name.get()
        self.create_face_folder()
        self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)



    # Mkdir for saving photos and csv
    def pre_work_mkdir(self):
        # Create folders to save face images and csv
        if os.path.isdir(self.path_photos_from_camera):
            pass
        else:
            os.mkdir(self.path_photos_from_camera)


    def check_existing_faces_cnt(self):
        if os.listdir("data/data_faces_from_camera/"):
            # Get the order of latest person
            person_list = os.listdir("data/data_faces_from_camera/")
            person_num_list = []

            for person in person_list:
                try:
                    person_order = int(person.split('_')[1].split('_')[0])
                    person_num_list.append(person_order)
                except ValueError:
                    # Handle folders with non-integer names here (skip them)
                    pass

            if person_num_list:
                self.existing_faces_cnt = max(person_num_list)
            else:
                self.existing_faces_cnt = 0

        # Start from person_1
        else:
            self.existing_faces_cnt = 0

    # Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        #  Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

        self.label_fps_info["text"] = str(self.fps.__round__(2))

    def create_face_folder(self):
        #  Create the folders for saving faces
        self.existing_faces_cnt += 1
        if self.input_name_char:
            self.current_face_dir = self.path_photos_from_camera + \
                                    "person_" + str(self.existing_faces_cnt) + "_" + \
                                    self.input_name_char
        else:
            self.current_face_dir = self.path_photos_from_camera + \
                                    "person_" + str(self.existing_faces_cnt)
        os.makedirs(self.current_face_dir)
        self.log_all["text"] = "\"" + self.current_face_dir + "/\" created!"
        logging.info("\n%-40s %s", "Create folders:", self.current_face_dir)

        self.ss_cnt = 0  #  Clear the cnt of screen shots
        self.face_folder_created_flag = True  # Face folder already created

    def get_frame(self):
        try:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                frame = cv2.resize(frame, (640,480))
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except:
            print("Error: No video input!!!")

    #  Main process of face detection and saving
    def process(self):
        ret, self.current_frame = self.get_frame()
        faces = detector(self.current_frame, 0)
        # Get frame
        if ret:
            self.update_fps()
            self.label_face_cnt["text"] = str(len(faces))
            #  Face detected
            if len(faces) != 0:
                #   Show the ROI of faces
                for k, d in enumerate(faces):
                    self.face_ROI_width_start = d.left()
                    self.face_ROI_height_start = d.top()
                    #  Compute the size of rectangle box
                    self.face_ROI_height = (d.bottom() - d.top())
                    self.face_ROI_width = (d.right() - d.left())
                    self.hh = int(self.face_ROI_height / 2)
                    self.ww = int(self.face_ROI_width / 2)

                    # If the size of ROI > 480x640
                    if (d.right() + self.ww) > 640 or (d.bottom() + self.hh > 480) or (d.left() - self.ww < 0) or (
                            d.top() - self.hh < 0):
                        self.label_warning["text"] = "OUT OF RANGE"
                        self.label_warning['fg'] = 'red'
                        self.out_of_range_flag = True
                        color_rectangle = (255, 0, 0)
                    else:
                        self.out_of_range_flag = False
                        self.label_warning["text"] = ""
                        color_rectangle = (255, 255, 255)
                    self.current_frame = cv2.rectangle(self.current_frame,
                                                       tuple([d.left() - self.ww, d.top() - self.hh]),
                                                       tuple([d.right() + self.ww, d.bottom() + self.hh]),
                                                       color_rectangle, 2)
            self.current_frame_faces_cnt = len(faces)

            # Convert PIL.Image.Image to PIL.Image.PhotoImage
            img_Image = Image.fromarray(self.current_frame)
            img_PhotoImage = ImageTk.PhotoImage(image=img_Image)
            self.label.img_tk = img_PhotoImage
            self.label.configure(image=img_PhotoImage)

        # Refresh frame
        self.win.after(20, self.process)

    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()

        self.process()
        self.win.mainloop()

def main():
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register()
    Face_Register_con.run()


def extract_faces(img):
    try:
        if img.shape != (0, 0, 0):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_points = face_detector.detectMultiScale(gray, 1.3, 5)
            return face_points
        else:
            return []
    except:
        return []


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('data/data_faces_from_camera/')

    for user in userlist:
        for imgname in os.listdir(f'data/data_faces_from_camera//{user}'):
            img = cv2.imread(f'data/data_faces_from_camera//{user}/{imgname}')
            resized_face = cv2.resize(img, (180, 180))

            # Check if the image is color or grayscale
            if len(resized_face.shape) == 3:  # Color image
                faces.append(resized_face.reshape(-1))  # Flatten the 3D array
            else:  # Grayscale image
                faces.append(resized_face.ravel())

            labels.append(user)

    faces = np.array(faces)
    labels = np.array(labels)

    # Assuming faces contains color images (3 channels), use faces.reshape(len(faces), -1) to flatten each image
    faces_flattened = faces.reshape(len(faces), -1)
    print('faces_flattened',faces_flattened)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces_flattened, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


def run_gui_logic():
    face_register = Face_Register()
    face_register.run()


@app.route('/')
def new():
    return render_template('home.html')


# @app.route("/start")
# def start_attendance():
#     # Run the face recognition code when the "/start" route is accessed
#     face_recognizer.run()
#     # return "Attendance recording is now running."
#     return render_template('page.html')
login_time = None
logout_time = None


@app.route("/start")
def start_attendance():
    global login_time
    # Run the face recognition code when the "/video_feed" route is accessed
    face_recog.run()

    # Get the logout time`
    login_time = datetime.datetime.now().strftime("%H:%M:%S")
    print('login_time', login_time)
    # Pass the login time and logout time to the template
    return render_template('page.html', login_time=login_time)


@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirmpassword = request.form['confirmpassword']

        cur = mysql.connection.cursor()

        cur.execute("INSERT INTO admin (name, email, password, confirmpassword) VALUES (%s, %s, %s, %s)",
                    (name, email, password, confirmpassword))

        mysql.connection.commit()

        cur.close()

        return render_template('page.html')

    return render_template('admin.html')


@app.route('/newuser', methods=['GET', 'POST'])
def newuser():
    if request.method == 'POST':
        id = request.form['id']
        full_name = request.form['full_name']
        email = request.form['email']
        phone = request.form['phone']
        birth_date = request.form['birth_date']
        department = request.form['department']

        userfolder = f'data/data_faces_from_camera/{full_name}_{id}'
        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users(id, full_name, email,phone, birth_date,department) VALUES(%s,%s, %s,%s, %s,%s)", (id,full_name, email, phone, birth_date, department))
        mysql.connection.commit()
        cur.close()

        knn = joblib.load('static/face_recognition_model.pkl')

        if os.path.exists(userfolder):
            return render_template('newuser.html', mess=f'{full_name}_{id} is already registered.')

        os.makedirs(userfolder)

        cap = cv2.VideoCapture(0)
        i = 0
        while i < 30:
            ret, frame = cap.read()
            faces = extract_faces(frame)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/30', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                            cv2.LINE_AA)

                # Check if the new face is similar to any registered face
                new_face = cv2.resize(frame[y:y + h, x:x + w], (180, 180))
                is_match = False

                for registered_face in registered_faces:
                    # Use the KNN model to predict if it's the same person
                    predicted_user = knn.predict(new_face.reshape(1, -1))
                    if predicted_user == registered_face:
                        is_match = True
                        break

                if is_match:
                    cv2.putText(frame, 'Face Already Registered!', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 0, 255), 2, cv2.LINE_AA)
                else:
                    if i < 30:
                        name = f'{full_name}_{i}.jpg'
                        cv2.imwrite(os.path.join(userfolder, name), frame[y:y + h, x:x + w])
                    i += 1

            cv2.imshow('Adding new User', frame)
            if cv2.waitKey(1) == 27 or i >= 30:
                break

        cap.release()
        cv2.destroyAllWindows()
        # Add the new user to the list of registered faces
        # registered_faces.append(newusername)
        registered_faces.append(f'{full_name}_{id}')
        return redirect(url_for('page'))
        print('Training Model')
        train_model()

        return redirect(url_for('page'))

    return render_template('newuser.html')


@app.route('/page')
def page():
    return render_template('page.html')


@app.route('/atten')
def atten():
    return render_template('report.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/retrain_model', methods=['GET'])
def retrain_model():

    knn = joblib.load('static/face_recognition_model.pkl')

    # Initialize empty lists for new data
    new_faces = []
    new_labels = []

    # Loop through the registered users
    for user_folder in os.listdir("data/data_faces_from_camera/"):
        user_path = os.path.join("data/data_faces_from_camera/", user_folder)
        for imgname in os.listdir(user_path):
            img = cv2.imread(os.path.join(user_path, imgname))
            resized_face = cv2.resize(img, (180,180))
            new_faces.append(resized_face.ravel())
            new_labels.append(user_folder)

    # Convert new_faces and new_labels to NumPy arrays
    new_faces = np.array(new_faces)

    # Retrain the model with the new dataset
    knn.fit(new_faces, new_labels)

    # Save the retrained model
    joblib.dump(knn, 'static/face_recognition_model.pkl')

    return "Model retraining completed successfully!"


@app.route('/run_main', methods=['POST'])
def run_face_registration():
    if request.method == 'POST':
        # Create an instance of the Face_Register class
        face_register = Face_Register()

        # Call the methods to perform the face registration
        face_register.pre_work_mkdir()  # Create necessary folders
        face_register.check_existing_faces_cnt()  # Check existing face counts
        face_register.GUI_get_input_name()  # Get input name (you might want to get this from the HTML form)
        face_register.run()
        # Redirect to the /newuser route after face registration is completed
        return redirect(url_for('newuser'))


def store_attendance(name, time,date):
    date = date.today().strftime("%d-%B-%Y")
    conn = sqlite3.connect('maindb.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO login_atten(name,time,date) VALUES ( ?, ?, ?)",
                   (name, time,date))
    conn.commit()
    conn.close()


def fetch_attendance():
    conn = connection.MySQLConnection(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port
    )
    cursor = conn.cursor(dictionary=True)

    # Fetch attendance data from the database for today
    # Implement code to retrieve attendance data
    cursor.execute("SELECT * FROM login_atten WHERE date = CURDATE()")
    attendance_data = cursor.fetchall()

    conn.close()

    return attendance_data


from mysql.connector import connection


host = "localhost"
user = "root"
password = "Pooja@12345"
database = "maindb"
port = 3307


@app.route("/video_feed")
def video_feed():
    global logout_time
    # Run the face recognition code when the "/video_feed" route is accessed
    face_recog.run()

    # Get the logout time`
    logout_time = datetime.datetime.now().strftime("%H:%M:%S")
    print('logout_time',logout_time)
    # Pass the login time and logout time to the template
    return render_template('page.html', logout_time=logout_time)


@app.route('/todays_attendance')
def show_attendance():
    # Create a connection
    conn = connection.MySQLConnection(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port
    )

    # Create a cursor
    cursor = conn.cursor(dictionary=True)

    # Fetch all rows from the atten table
    cursor.execute("SELECT * FROM login_atten")
    all_rows = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    conn.close()

    return render_template('attendance.html', attendance_data=all_rows, login_time=login_time, logout_time=logout_time)


if __name__ == '__main__':
    app.run(port=5001, debug=True)


from google.colab import drive
drive.mount('/content/drive')
import cv2
import pandas as pd
from google.colab.patches import cv2_imshow

def Face_detection(frame,folder_name,live, video_name, frame_name, results_df):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        df2 = pd.DataFrame ({'Folder Name':folder_name,'Video Name': video_name, 'Frame Name': frame_name, 'X': x.astype(str), 'Y': y.astype(str), 'Width': w.astype(str), 'Height': h.astype(str),'Liveness':live}, index=[0])
        results_df = pd.concat([results_df, df2], ignore_index=True)
    return frame, results_df

def video_processing(video_path, results_df):
    cap = cv2.VideoCapture(video_path)
    path_parts = video_path.split('/')

    folder_name = path_parts[-2]
    
    video_name = video_path.split('/')[-1]
    live="0"
    if video_name=="1.avi" or video_name=="2.avi" or video_name=="HR_1.avi" or video_name=="HR_4.avi":
      live="1"
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video Name   "+video_name+"   "+folder_name)
  
    for frame_number in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, results_df = Face_detection(frame,folder_name,live, video_name, f'Frame_{frame_number}', results_df)

    cap.release()
    return results_df

results_df = pd.DataFrame(columns=['Folder Name','Video Name', 'Frame Name', 'X', 'Y', 'Width', 'Height','Liveness'])
path="/content/drive/MyDrive/CASIA_faceAntisp/train_release/"
video_files=[]
for i in range (1,21):
  for j in range (1,9):
    new_path=path+str(i)+"/"+str(j)+".avi"
    video_files.append(new_path)
  for j in range (1,5):
    new_path=path+str(i)+"/HR_"+str(j)+".avi"
    video_files.append(new_path)

for video_file in video_files:
    results_df=video_processing(video_file, results_df)

print(results_df)
results_df.to_csv('/content/drive/MyDrive/CASIA_faceAntisp/data.csv', index=False)

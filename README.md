# face-recognition
Use OpenCV to recognize faces from a video file

## Description of files
This project uses OpenCV and deep learning to:
1. Detect Faces
2. Compute 128-d face embeddings to quantify a face
3. Train a SVM on top of the embeddings
4. Recognize faces in videos

The [dataset](dataset) has the images of the cast from the Jurassic Park movie. The folder name will be used by the machine to label the faces once it recognizes them.

The [face detection model](face_detection_model) contains a pre-trained Caffe deep learning model provided by OpenCV to detect faces.
It detects and localises the faces in the video.

The [output](output) file has the output pickle files:
- [embeddings.pickle](output/embeddings.pickle) has the serialized facial embeddings computed by [extract_embeddings.py](extract_embeddings.py).
- [le.pickle](output/le.pickle) has label encoding for the people that the model can recognize.
- [recognizer.pickle](output/recognizer.pickle) is the Linear Support Vector Model (SVM) which recognizes the faces.

[openface_nn4.small2.v1.t7](openface_nn4.small2.v1.t7) is a Torch deep learning model which produces the 128-D facial embeddings.

[train_model.py](train_model.py) is the script which trains our Linear SVM model.

[recognize_video_file.py](recognize_video_file.py) uses imultils' VideoFileStream to open the video file and then uses the output from the embeddings and training model to recognize the faces.

**imutils** is a very useful package which has a series of functions to make basic image processing operations like resizing and rotation with OpenCV and Python. You can find it [here](https://github.com/jrosebr1/imutils)

Thanks to Adrian Rosebrick's tutorial on [PyImageSearch](https://www.pyimagesearch.com/) which helped me get familiar with face detection and face recognition.

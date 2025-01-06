# Stress-Detection-in-Real-Time
A python script using OpenCV, Tensorflow, Keras and Deepface to detect people's levels of stress in real time.


### How to use: 
Install the following packages:
```bash
pip install deepface
pip install tf_keras
pip install opencv-python
```
Alternatively if you get the error "Cannot find the file specified" when running the script, manually reinstall the file facial_expression_model_weights_h5 located at users/[...]/.deepface/weights/  from https://github.com/serengil/deepface_models/releases/ .
If that doesn't work, then do the following commands

```bash
pip uninstall deepface
git clone https://github.com/serengil/deepface.git
cd deepface
pip install -e .
```

And then do the manually reinstallation of the file specified above. In any other case, this step will provide a better error code, so if you still encounter issues, feel free to reach out!

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.optimizers import Adam

from flask import*
import os

app=Flask(__name__)
UPLOAD_FOLDER='static/data'
app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER

@app.route('/')
@app.route("/bulk",methods=['GET','POST'])

def home():
    return render_template("deep_fake.html")
def bulk():
    if request.method=="POST":
        file=request.files['data']
        na=file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'],file.filename))
        
        '''with open("F:\ml\dc_model.json", "r") as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)'''

        loaded_model.load_weights("F:\ml\deepfake_detection_modelupdated.h5")

        # Compile the loaded model (necessary for predictions)
        loaded_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

        # Function to preprocess an image for prediction
        def preprocess_image(image_path):
            img = load_img(image_path, target_size=(150, 150))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            return img_array

        # Test image path
        test_image_path = r"static/data/"+na # Replace with the path of the test image

        # Preprocess the test image for prediction
        test_image = preprocess_image(test_image_path)

        # Make predictions on the test image
        prediction = loaded_model.predict(test_image)

        # Classify the prediction (assuming binary classification)
        os.remove(test_image_path)
        if(prediction[0][0]>=0.5):
            return render_template("deep_fake.html",name="REAL")
        else:
            return render_template("deep_fake.html",name="FAKE")
        
    else:
        return render_template("deep_fake.html",name="")
    
if __name__=='__main__':
    app.run(debug=True)

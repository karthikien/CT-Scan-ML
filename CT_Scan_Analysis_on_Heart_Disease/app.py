from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename 
from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np
import cv2 
from pymongo import MongoClient
import os

connection_string = os.environ.get('MONGODB_CONNECTION_STRING')
app = Flask(__name__)

UPLOAD_FOLDER = 'static/images/uploads' # Where to store uploaded images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'} 
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_path = 'data/new_data/trained_model.keras'  # Make sure this path is correct
model = load_model(model_path)

client = MongoClient('mongodb+srv://heart:12345qwerty@heartpredict.qdnoyei.mongodb.net/?retryWrites=true&w=majority&appName=heartpredict')
db = client['heart_risk_database']
collection = db['heart_risk_collection']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def resize_ct_image(image_path, output_path, target_size=(224, 224)):
    try:
        # Read the CT image
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Resize the image to the target size
        resized_img = cv2.resize(img, target_size)

        # Save the resized image
        cv2.imwrite(output_path, resized_img)

        print(f"Image resized and saved to {output_path}")

        return resized_img

    except Exception as e:
        print(f"Error resizing image: {e}")
        return None

@app.route('/detect', methods=['POST'])
def detect():
    if 'image_file' not in request.files:
        return redirect('/')  # Redirect to the homepage or display an error message

    file = request.files['image_file']
    if file.filename == '':
        return redirect('/')  # Redirect to the homepage or display an error message

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Resize and preprocess the image
        resized_img = resize_ct_image(filepath, "static/images/resized_image.jpg", target_size=(224, 224))

        if resized_img is not None:
            # Convert grayscale image to 3 channels
            img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)

            # Normalize pixel values to the range [0, 1]
            img = img / 255.0

            # Add batch dimension
            img = np.expand_dims(img, axis=0)

            # Make the prediction
            prediction = model.predict(img)

            # Interpret and get risk_level
            if prediction >= 0.30:
                risk_level = "You have a Diseased Heart!!" 
            else:
                risk_level = "Your Heart is Healthy" 

            document = {
                'filename': filename,
                'risk_level': risk_level
            }
            collection.insert_one(document)


            return render_template('result.html', risk_level=risk_level, image_filename=filename) 
        else:
            return redirect('/')  # Redirect to the homepage or display an error message
    else:
        return redirect('/')  # Redirect to the homepage or display an error message


def calculate_risk_level(calcium_score):
    if calcium_score == 0:
        return "Very low risk"
    elif 1 <= calcium_score <= 50:
        return "Low risk"
    elif 51 <= calcium_score <= 250:
        return "Moderate risk"
    elif 251<= calcium_score <= 600:
        return "High risk"
    elif 601<= calcium_score <= 999:
        return "Very high risk"
    else :
        return "Dangerous!!!Seek Help Immediatly"
    
@app.route('/result', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the form data
        age = int(request.form['Age'])
        calcium_score = int(request.form['Calcium_score'])
        epicardial_volume = int(request.form['Epicardial_Tissue_Volume'])
        pericardial_volume = int(request.form['Pericardial_Tissue_Volume'])
        cardiac_fats = int(request.form['Sum_of_Cardiac_Fats'])
        
        # Calculate risk level based on calcium score
        risk_level = calculate_risk_level(calcium_score)

        prediction_data = {
            'age': age,
            'calcium_score': calcium_score,
            'epicardial_volume': epicardial_volume,
            'pericardial_volume': pericardial_volume,
            'cardiac_fats': cardiac_fats,
            'risk_level': risk_level
        }
        collection.insert_one(prediction_data)
        
        # Render the result template with the calculated risk level
        return render_template('result.html', risk_level=risk_level)

    # If it's a GET request, just render the form template
    return render_template('index.html')

@app.route('/')
def index():
    return render_template('index.html') 
@app.route('/riskfactor')
def riskfactor():
    return render_template('riskfactor.html') 

if __name__ == '__main__':
    app.run(debug=True)
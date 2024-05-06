import cv2
import numpy as np
from keras.models import load_model

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


# Load the trained model
model_path = "data/new_data/trained_model.keras"
model = load_model(model_path)

# Load and preprocess the inserted image
inserted_image_path = "Insert here/IMG-0002-00001.jpg"  # Replace with the actual path

resized_img = resize_ct_image(inserted_image_path, "data/inserted_resized_image.jpg", target_size=(224, 224))

if resized_img is not None:
    # Convert grayscale image to 3 channels
    img = cv2.cvtColor(resized_img, cv2.COLOR_GRAY2RGB)

    # Normalize pixel values to the range [0, 1]
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Make the prediction
    prediction = model.predict(img)

    # Display the result
    print(f"Prediction Value: {prediction[0][0]}")

    if prediction >= 0.5:
        print("The inserted image is predicted as diseased.")
    else:
        print("The inserted image is predicted as healthy.")

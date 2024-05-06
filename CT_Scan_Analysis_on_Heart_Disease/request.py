import requests

# Define the URL of the Flask server
url = 'http://localhost:5000/predict'

# Define the path to the image file
image_path = 'data/Insert here/'  # Update with the actual path to your image

# Open the image file
with open(image_path, 'rb') as file:
    # Create a dictionary to contain the file data
    files = {'file': file}

    # Send a POST request to the Flask server with the image file
    response = requests.post(url, files=files)

# Check if the request was successful
if response.status_code == 200:
    result = response.json()['result']
    print(f'Prediction result: {result}')
else:
    print('Error occurred during prediction')

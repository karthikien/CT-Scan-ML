import os
import cv2
import pydicom
import numpy as np
from keras.applications import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Define paths
healthy_path = "data/healthy"
disease_path = "data/disease"
output_dir = "data/new_data"

def load_image(path):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None and not img.size == 0:
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  
            img = img / 255.0
            return img
        dcm = pydicom.dcmread(path)
        img = dcm.pixel_array
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  
        return img
    except Exception as e:
        print(f"Error loading image at {path}: {e}")
        return None

def load_and_process_images(data_path, output_dir, labels):
    images = []
    for patient_dir in os.listdir(data_path):
        patient_path = os.path.join(data_path, patient_dir)
        if os.path.isdir(patient_path):
            
            for study_dir in os.listdir(patient_path):
                study_path = os.path.join(patient_path, study_dir)
                if os.path.isdir(study_path):
                    for filename in os.listdir(study_path):
                        if filename.endswith('.dcm'):
                            img_path = os.path.join(study_path, filename)
                            try:
                                image = load_image(img_path)
                                if image is not None:
                                    output_filename = f"{patient_dir}_{study_dir}_{filename[:-4]}.jpg"
                                    output_path = os.path.join(output_dir, output_filename)
                                    cv2.imwrite(output_path, image)
                                    print(f"Saved to {output_path}")
                                    images.append(image)
                            except Exception as e:
                                print(f"Error loading image {img_path}: {e}")
    return images, labels

# Process healthy images
os.makedirs(output_dir, exist_ok=True)
healthy_images = []
for patient_dir in os.listdir(healthy_path):
    patient_path = os.path.join(healthy_path, patient_dir)
    if os.path.isdir(patient_path): 
        for study_dir in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study_dir)
            if os.path.isdir(study_path):  
                for filename in os.listdir(study_path):
                    if filename.endswith('.dcm'):
                        img_path = os.path.join(study_path, filename)
                        print("Processing:", img_path)
                        try:
                            image = load_image(img_path)
                            healthy_images.append(image)
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
    else:
        print(f"Skipping non-directory: {patient_path}")
healthy_labels = [0] * len(healthy_images)

# Process disease images
disease_images = []
for patient_dir in os.listdir(disease_path):
    patient_path = os.path.join(disease_path, patient_dir)
    if os.path.isdir(patient_path):  
        for study_dir in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study_dir)
            if os.path.isdir(study_path):  
                for filename in os.listdir(study_path):
                    if filename.endswith('.dcm'):
                        img_path = os.path.join(study_path, filename)
                        print("Processing:", img_path)
                        try:
                            image = load_image(img_path)
                            disease_images.append(image)
                        except Exception as e:
                            print(f"Error loading image {img_path}: {e}")
    else:
        print(f"Skipping non-directory: {patient_path}")

disease_labels = [1] * len(disease_images)

# Combine data and labels
X = healthy_images + disease_images
y = healthy_labels + disease_labels

# Data split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

print("Number of healthy images:", len(healthy_images))
print("Number of disease images:", len(disease_images))

# Create generators
train_generator = datagen.flow(np.array(X_train), np.array(y_train), batch_size=32, shuffle=True)
val_generator = datagen.flow(np.array(X_val), np.array(y_val), batch_size=32, shuffle=False)

# Load pre-trained ResNet50 and exclude top layers
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Add global average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# Add a classification layer with sigmoid activation for binary output
predictions = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, epochs=10, validation_data=val_generator)

feature_extraction_layer = 'global_average_pooling2d'  # Penultimate layer
feature_extractor_model = Model(inputs=model.input,
                                outputs=model.get_layer(feature_extraction_layer).output)

# Extract and save features for healthy images
features_train_healthy = feature_extractor_model.predict(np.array(healthy_images))
np.save(os.path.join(output_dir, "healthy_features.npy"), features_train_healthy)
print(f"Healthy features saved to {os.path.join(output_dir, 'healthy_features.npy')}")

# Extract and save features for disease images
features_train_disease = feature_extractor_model.predict(np.array(disease_images))
np.save(os.path.join(output_dir, "disease_features.npy"), features_train_disease)
print(f"Disease features saved to {os.path.join(output_dir, 'disease_features.npy')}")

# Example: Extract features from training set
features_train = feature_extractor_model.predict(np.array(X_train))

# -------- Example Usage 1: Visualization --------
reduced_features = TSNE(n_components=2).fit_transform(features_train)
# ... Create a scatter plot using reduced_features and y_train 
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=y_train)
plt.title('Visualization of Extracted Features (TSNE)')
plt.xlabel('TSNE Dimension 1')
plt.ylabel('TSNE Dimension 2')
plt.colorbar(label='Heart Disease Class')  # Add a colorbar
plt.show()

# -------- Example Usage 2: Classification --------
clf = LogisticRegression(solver='liblinear')
clf.fit(features_train, y_train)
y_pred = clf.predict(feature_extractor_model.predict(np.array(X_val)))

# Print the classification report
class_report = classification_report(y_val, y_pred, target_names=['Healthy', 'Disease'])
print("Classification Report:\n", class_report)

# Create and plot the confusion matrix
cm = confusion_matrix(y_val, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Healthy', 'Disease'], yticklabels=['Healthy', 'Disease'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Save the trained model
model.save(os.path.join(output_dir, 'trained_model.keras'))
print(f"Trained model saved to {os.path.join(output_dir, 'trained_model.keras')}")

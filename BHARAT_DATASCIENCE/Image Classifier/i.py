import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# Directory paths to your dataset
train_directory = "C:\\Users\\Sakshi\\OneDrive\\Desktop\\BHARAT_DATASCIENCE\\Image Classifier\\dataset\\train"
test_directory = "C:\\Users\\Sakshi\\OneDrive\\Desktop\\BHARAT_DATASCIENCE\\Image Classifier\\dataset\\test"

# Load images using ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_ds = train_datagen.flow_from_directory(
    train_directory,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'  # Assuming binary classification (cats vs dogs)
)

validation_ds = test_datagen.flow_from_directory(
    test_directory,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

# Define your model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

# Display some images and their corresponding labels
class_names = ['Cat', 'Dog']

sample_images, sample_labels = next(validation_ds)
predicted_labels = model.predict(sample_images)

num_samples = min(9, len(sample_images))  # Number of samples to display
plt.figure(figsize=(10, 10))
for i in range(num_samples):
    plt.subplot(3, 3, i + 1)
    plt.imshow(sample_images[i])
    plt.title(f"Predicted: {class_names[int(np.round(predicted_labels[i]))]}\nActual: {class_names[int(sample_labels[i])]}")
    plt.axis('off')
plt.show()

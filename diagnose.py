import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import classification_report

# Load your saved model
model = tf.keras.models.load_model('optimized_cnn_model.h5')

# Setup data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    'C:/Users/Rohit Negi/Desktop/Projects/Major Project/Acute Lymphoblastic Leukemia dataset/Split_Data/test',
    target_size=(224, 224),
    batch_size=1,
    shuffle=False
)

# Print class distribution
class_counts = np.bincount(test_generator.classes)
class_labels = list(test_generator.class_indices.keys())

print("\nClass Distribution:")
for label, count in zip(class_labels, class_counts):
    print(f"{label}: {count} images ({count/len(test_generator.classes)*100:.2f}%)")

# Get predictions
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Print classification report
print("\nDetailed Classification Report:")
print(classification_report(test_generator.classes, predicted_classes,
                          target_names=class_labels, digits=4))
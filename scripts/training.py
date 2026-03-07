import os
import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Rescaling, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# fix import error
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from app.config import TRAIN_DIR, TEST_DIR, IMG_SIZE,BATCH_SIZE, EPOCHS, MODEL_SAVE_PATH

def build_and_train_model():
    print("🚀 Starting Model Training Pipeline...")

    # --- 2. Data Loading & Augmentation ---
    # We use ImageDataGenerator to automatically load images from folders
    # and scale the pixels from 0-255 down to 0-1.
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        zoom_range=0.1,
        horizontal_flip=True
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    val_generator = val_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    num_classes = len(train_generator.class_indices)
    print(f"🧠 Detected Classes: {train_generator.class_indices}")

    # --- 3. Build the CNN Architecture ---
    model = Sequential([
        # Block 1
        Input(shape=(48, 48, 1)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Classification Head
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # --- 4. Train the Model ---
    print(f"⏳ Training for {EPOCHS} epochs...")
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS
    )

    # --- 5. Save the Model ---
    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model successfully saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    build_and_train_model()
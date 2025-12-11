# ===============================
# train_model.py (Spark + CNN)
# ===============================

import os
from pyspark.sql import SparkSession
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# -----------------------------------------
# 🔥 Initialize Spark Session
# -----------------------------------------
spark = (
    SparkSession.builder
    .appName("MaskDetection_Training_Spark")
    .config("spark.ui.port", "4040")
    .getOrCreate()
)
print("✅ Spark session started successfully!")

# -----------------------------------------
# 🧠 Load dataset info using Spark
# -----------------------------------------
image_data = []
for root, dirs, files in os.walk("dataset"):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Folder name = label (e.g., 'with_mask', 'without_mask')
            image_data.append((file, os.path.basename(root)))

if not image_data:
    print("⚠️ No images found in 'dataset/' folder. Check your dataset path!")

df = spark.createDataFrame(image_data, ["filename", "label"])
print(f"✅ Loaded {df.count()} image records into Spark DataFrame.")
df.show(10)

# Cache the dataset in Spark memory
df.cache()
print("🧠 Dataset cached in Spark memory for monitoring.")

# -----------------------------------------
# 📊 Data preprocessing and augmentation
# -----------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2,  # 80% training, 20% validation
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    "dataset/",
    target_size=(150, 150),
    batch_size=16,
    class_mode="binary",
    subset="training",
)

val_generator = train_datagen.flow_from_directory(
    "dataset/",
    target_size=(150, 150),
    batch_size=16,
    class_mode="binary",
    subset="validation",
)

# -----------------------------------------
# 🧩 CNN Model
# -----------------------------------------
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# -----------------------------------------
# 🚀 Training
# -----------------------------------------
print("🚀 Training started...")
history = model.fit(train_generator, epochs=10, validation_data=val_generator)
print("✅ Training completed successfully!")

# -----------------------------------------
# 💾 Save model
# -----------------------------------------
os.makedirs("model", exist_ok=True)
model.save("model/mask_detector.h5")
print("✅ Model saved as model/mask_detector.h5")

# -----------------------------------------
# 📦 Log training summary in Spark
# -----------------------------------------
train_summary = [
    ("accuracy", float(history.history["accuracy"][-1])),
    ("val_accuracy", float(history.history["val_accuracy"][-1])),
    ("loss", float(history.history["loss"][-1])),
    ("val_loss", float(history.history["val_loss"][-1])),
]

summary_df = spark.createDataFrame(train_summary, ["metric", "value"])
summary_df.show()

os.makedirs("spark_storage", exist_ok=True)
summary_df.write.mode("overwrite").parquet("spark_storage/training_summary.parquet")
print("📊 Training summary saved to spark_storage/training_summary.parquet")

# -----------------------------------------
# 🏁 Stop Spark
# -----------------------------------------
spark.stop()
print("🧹 Spark session closed.")

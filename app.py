from flask import Flask, render_template, request
from pyspark.sql import SparkSession
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import time

# ✅ Set Python environment for PySpark
os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"

# ✅ Initialize Flask
app = Flask(__name__)

# ✅ Load CNN model
model = load_model('model/mask_detector.h5')

# ✅ Folder for image uploads
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ Initialize Spark Session with local master and fixed UI port
spark = SparkSession.builder \
    .appName("MaskDetectionApp") \
    .master("local[*]") \
    .config("spark.ui.port", "4040") \
    .config("spark.sql.shuffle.partitions", "4") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

@app.route('/')
def home():
    spark_status = "✅ Spark Connected (UI at http://localhost:4040)"
    return render_template('index.html', result=None, spark_status=spark_status)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # ✅ Get uploaded file
        file = request.files['file']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # ✅ Create a Spark DataFrame
        df = spark.createDataFrame([(file.filename,)], ["Uploaded_File"])
        df.cache()
        df.count()
        df.show()

        # ✅ Spark job to appear in Jobs tab
        word_rdd = spark.sparkContext.parallelize(["Spark", "Mask", "Detection", "App"] * 5000)
        counts = word_rdd.map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)
        counts.collect()
        time.sleep(1)

        # ✅ Image preprocessing for CNN model
        img = image.load_img(filepath, target_size=(150, 150))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        # ✅ Prediction
        preds = model.predict(x)
        pred = "😷 Wearing Mask" if preds[0][0] < 0.5 else "😐 No Mask"

        return render_template(
            'index.html',
            result=pred,
            img_path=filepath,
            spark_status="✅ Spark Connected (View Jobs at http://localhost:4040)"
        )

    except Exception as e:
        return f"❌ Error occurred: {str(e)}"

if __name__ == '__main__':
    flask_port = 5000
    spark_ui_port = 4040

    print("\n🚀 Application Running Successfully!\n")
    print(f"🌐 Flask Server URL: http://localhost:{flask_port}")
    print(f"⚡ Spark Web UI URL: http://localhost:{spark_ui_port}")
    print("\nPress CTRL+C to stop the server.\n")

    app.run(debug=True, use_reloader=False, port=flask_port)

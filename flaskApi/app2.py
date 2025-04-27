from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import shutil  
import sys
import base64
from io import BytesIO
from PIL import Image
from ecgAnalyser import analyze_ecg
# sys.path.append(os.path.abspath('../newNotebook'))
# from ecgAnalyser import analyze_ecg  
import numpy as np

import cloudinary
import cloudinary.uploader
import cloudinary.api

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Set up Cloudinary config
cloudinary.config(
    cloud_name = os.getenv('CLOUD_NAME'), 
    api_key = os.getenv('API_KEY'), 
    api_secret = os.getenv('API_SECRET')
)


# Initialize the Flask application
app = Flask(__name__)

CORS(app, origins=["http://localhost:3000"])
# CORS(app) 
# model = joblib.load("model.pkl")

@app.route('/')
def index():
    return "ML API is running!"

@app.route('/demo',methods=['POST'])
def demo():
    print('data is received')
    data=request.get_json()
    print(data)
    return jsonify({'result':'Hey im from flask'})

# @app.route('/upload', methods=['POST'])
# def upload_file():

@app.route('/predict', methods=['POST'])
def predict():
    print('inside upload file')
    files = request.files.getlist('files')
    id = request.form.get('id')
    print(id)
    print(files)

    upload_folder = os.path.join('./uploads', id)
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
        print(f"Created directory {upload_folder}")
    else:
        print(f"Directory {upload_folder} already exists")

    saved_files = []
    recordName = None
    for f in files:
        if not recordName:
            recordName, _ = os.path.splitext(f.filename)
        filepath = os.path.join(upload_folder, f.filename)
        f.save(filepath)
        saved_files.append(f.filename)
    results = None
    cloudinary_urls = []  # To store URLs of the uploaded images

    try:
        model_input_path = os.path.join(upload_folder, recordName)  # Folder + filename without extension
        print('model input path is ')
        print(model_input_path)
        absolute_path = os.path.abspath(model_input_path)
        results = analyze_ecg(record_path=absolute_path, with_xai=True)
        img_base64 = results['images']

        print(img_base64.keys())

        # Upload images to Cloudinary
        for img_name, img_data in img_base64.items():
            # Convert the image binary data to a file-like object
            img_bytes = base64.b64decode(img_data)

            # Create a BytesIO object
            img_io = BytesIO(img_bytes)

            # Upload to Cloudinary
            upload_result = cloudinary.uploader.upload(img_io, folder="ecg_images")
            cloudinary_urls.append(upload_result['secure_url'])  # Store the Cloudinary URL

        # Clean up by removing the uploaded folder
        shutil.rmtree(upload_folder)
        print(f"Deleted folder {upload_folder} after processing.")
    except Exception as e:
        print(f"Error processing files: {e}")

    # Return response with image URLs from Cloudinary
    return jsonify({
        'message': 'Files and data received successfully',
        'received_files': saved_files,
        'image_urls': cloudinary_urls  # Return the URLs
    }), 200

# @app.route('/predict', methods=['POST'])
# def predict():
#     print('inside upload file')
#     files = request.files.getlist('files')
#     id=request.form.get('id')
#     print(id)
#     print(files)

#     upload_folder = os.path.join('./uploads', id)
#     if not os.path.exists(upload_folder):
#         os.makedirs(upload_folder)  
#         print(f"Created directory {upload_folder}")
#     else:
#         print(f"Directory {upload_folder} already exists")

#     saved_files = []
#     recordName=None
#     for f in files:
#         if(not recordName):
#             recordName, _ = os.path.splitext(f.filename)
#         filepath = os.path.join(upload_folder, f.filename)
#         f.save(filepath)
#         saved_files.append(f.filename)
#     results=None
#     try:
#         # results = analyze_ecg(record_path='D:\\DeepXAI\\datasets\\mit-bih\\121', with_xai=True)
#         model_input_path = os.path.join(upload_folder, recordName)  # Folder + filename without extension
#         print('model input path is ')
#         print(model_input_path)
#         absolute_path = os.path.abspath(model_input_path)
#         results= analyze_ecg(record_path=absolute_path,with_xai=True)
#         img_base64 = results['images']
        
#         print(img_base64.keys())
        
#         shutil.rmtree(upload_folder)
#         print(f"Deleted folder {upload_folder} after processing.")
#     except Exception as e:
#         print(f"Error deleting folder {upload_folder}: {e}")

#     return jsonify({
#             'message': 'Files and data received successfully',
#             'received_files': saved_files,
#         }), 200
        


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
    # app.run(debug=True)


            # 'info': text_data
        # if file.filename == '':
        #     return jsonify({"message": "No selected file"}), 400
        # # Process the file (e.g., save or analyze)
        # # Here, you can save it, or process it directly.
        # file.save(f"./uploads/{file.filename}")  # Example saving file

        # return jsonify({"message": f"File {file.filename} uploaded successfully!"}), 200
    # except Exception as e:
    #     print(e)
    #     return jsonify({"message": "File upload failed!"}), 500

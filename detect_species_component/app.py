from flask import Flask, render_template, request, jsonify, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import json
from PIL import Image
import json
import numpy as np
from datetime import timedelta
import yolov4
import yaml
import shutil
from azure.storage.blob import ContainerClient, BlobClient, BlobLeaseClient, BlobServiceClient, ContentSettings

# path to input data
IMAGES_PATH = './input_images/'
VIDEO_PATH = './input_videos/'

TEMP_VIDEO_DATA = './videoData/'
CONTAINER_CONFIG_PATH = '/cfg/config.yaml'

ALLOWED_EXTENSIONS = set(['png', 'PNG', 'jpg', 'jpeg',  'JPG', 'bmp'])
VIDEO_EXTENSIONS = set(['mp4', 'avi'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
def allowed_vfile(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in VIDEO_EXTENSIONS

def load_config():
    dir_root = os.path.dirname(os.path.abspath(__file__))
    with open(dir_root + CONTAINER_CONFIG_PATH, "r") as yamlfile:
        return yaml.load(yamlfile, Loader=yaml.Loader )
    
app = Flask(__name__)

# app route - Upload multiple images for species detection component
@app.route('/uploadimages', methods=['POST'])
def uploadMult():
    files = request.files.getlist('image')
    print('images uploaded')
    data = []
    for f in files:
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "File type exception !"})
        # t is the name of the obtained picture
        t = f.filename
        # division - take the file name without .jpg
        filename_ = t.split('.')[0]
        user_input = request.form.get("name")
        # project main directory
        basepath = os.path.dirname(__file__)
        # save images temporarily
        try:
            if not os.path.exists(IMAGES_PATH):
                os.makedirs(IMAGES_PATH)
        except OSError:
            print ('Error: Failed to create folder')
        upload_path = os.path.join(basepath, IMAGES_PATH, secure_filename(f.filename))
        f.save(upload_path)
        # detect species in specific image
        print('detect species started')
        lab, img, loc, res = yolov4.yolo_detect(pathIn=upload_path)
        
        # filter images doesnt contain bluespotted object
        if res['counts'] > 0:
            #append the json result of all images
            data.append(res)
            
    shutil.rmtree(IMAGES_PATH)
    final_result = jsonify(data)
    return final_result


# app route - Copy blob image
@app.route('/copyBlobImage', methods=['POST'])
def blob_copy():
    c = load_config()
    json_info = request.get_json(force=True)
    url =  json_info['url']
    fileName = json_info['fileName']
    directoryPath = json_info['individual_ID']

    destination_blob_name = directoryPath + '/' + fileName

    client = BlobServiceClient.from_connection_string(c["azure_storage_connectionstring"])
    new_blob = client.get_blob_client(c["ident_container_name"],destination_blob_name)
    res = new_blob.start_copy_from_url(url)
            
    final_result = jsonify(res)
    return final_result

    
# app route - Upload video for species detection component
@app.route("/uploadVideo", methods=['POST'])
def uploadVideo():
    id = request.form['id']
    f = request.files['file']
    if not (f and allowed_vfile(f.filename)):
        return jsonify({"error": 1001, "msg": "File type exception !"})
    
    # t is the name of the obtained picture
    t = f.filename
    # division - take the file name without .jpg
    filename_ = t.split('.')[0]
    user_input = request.form.get('name')
    # project main directory
    basepath = os.path.dirname(__file__)
    # save video temporarily
    try:
        if not os.path.exists(VIDEO_PATH):
            os.makedirs(VIDEO_PATH)
    except OSError:
        print ('Error: Failed to create folder')
    upload_path = os.path.join(basepath, VIDEO_PATH, secure_filename(f.filename))
    f.save(upload_path)
    
    # load video from path:
    cap = cv2.VideoCapture(VIDEO_PATH + t)
    
    def load_config():
        dir_root = os.path.dirname(os.path.abspath(__file__))
        with open(dir_root + CONTAINER_CONFIG_PATH, "r") as yamlfile:
            return yaml.load(yamlfile, Loader=yaml.Loader )
    
    def get_file(dir):
        for entry in os.scandir(dir):
                if entry.is_file() and not entry.name.startswith('.'):
                    yield entry
                    
    # uploading images to blob storage
    def upload_images(files, connection_string, container_name):
        print("Uploading frames to blob storage")
        container_client = ContainerClient.from_connection_string(connection_string, container_name)
        for file in files:
            blob_client = container_client.get_blob_client(file.name)
            with open(file.path, "rb") as data:
                uploaded = blob_client.upload_blob(data)
                
    # uploading video to blob storage           
    def upload_video(files, connection_string, container_name):
        print("Uploading video to blob storage")
        container_client = ContainerClient.from_connection_string(connection_string, container_name)
        for file in files:
            blob_client = container_client.get_blob_client(file.name)
            with open(file.path, "rb") as data:
                my_content_settings = ContentSettings(content_type='video/mp4')
                uploaded = blob_client.upload_blob(data, overwrite=True, content_settings=my_content_settings)

    config = load_config()
    video = get_file(VIDEO_PATH)
    upload_video(video, config["azure_storage_connectionstring"], config["videos_container_name"])
    
    try:
        if not os.path.exists(TEMP_VIDEO_DATA):
            os.makedirs(TEMP_VIDEO_DATA)
    except OSError:
        print ('Error: Failed to create folder')
    
    current_frame = 0
    i = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        if i%58==0:
            # create new frame
            print("created frame {}".format(current_frame))
            name = './videoData/{}'.format(filename_) + "_" + str(current_frame) + '.jpg'
            cv2.imwrite(name, frame)
            current_frame += 1 
        i+=1
    data = []
    res=0    
    
    # detection for all frames created from the video
    for filename in os.listdir(TEMP_VIDEO_DATA):
        if filename.endswith('.jpg'):
            name = TEMP_VIDEO_DATA + filename
            
            # detect species in specific image
            lab, img, loc, res = yolov4.yolo_detect(pathIn=name)
            
            # filter images doesnt contain bluespotted object
            if res['counts']==0:
                os.remove(TEMP_VIDEO_DATA + filename)
            else: data.append(res)
                
    # upload all videos frames to blob storage
    picture = get_file(TEMP_VIDEO_DATA)
    upload_images(picture, config["azure_storage_connectionstring"], "encountersraw/{}".format(id))
        
    shutil.rmtree(TEMP_VIDEO_DATA) 
    os.remove(VIDEO_PATH + t)
    final_result = jsonify(data)
    return final_result
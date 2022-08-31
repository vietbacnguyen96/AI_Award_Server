import os
from sqlite3 import Time, Timestamp
import numpy as np
import cv2
from PIL import Image
from numpy import dot, sqrt
import re

from flask import Response, request, jsonify, redirect, url_for, send_file
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms

import hnswlib

from app.backbone import Backbone
from app.vision.ssd.config.fd_config import define_img_size
from app.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor

from app.arcface_torch.backbones import get_model
from app.arcface_torch.branch_util import *
from app.arcface_torch.mark_detector import MarkDetector
# from app.arcface_torch.pose_estimator import PoseEstimator


from sqlalchemy import func
from sqlalchemy.sql import text
from create_app import create_app, db, DefineImages, People, Timeline, Users, verify_pass, login_manager
from flask_login import (
    current_user,
    login_user,
    logout_user,
    login_required
)
from flask_migrate import Migrate
from flask_socketio import SocketIO
from pubsub import pub

import base64
import requests
import uuid

app = create_app()
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'
Migrate(app, db)
socketio = SocketIO(app)

def listener(uid, message):
    socketio.emit(uid,{'message': message})
pub.subscribe(listener, 'face_vkist')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class_names = [name.strip() for name in open('app/detect_RFB_640/voc-model-labels.txt').readlines()]
candidate_size = 1000
threshold = 0.7
input_img_size = 640
define_img_size(input_img_size)
model_path = "app/detect_RFB_640/version-RFB-640.pth"
net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=device)
predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=device)
net.load(model_path)

input_size=[112, 112]
transform = transforms.Compose(
        [
            transforms.Resize(
                [int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)],
            ),  # smaller side resized
            transforms.CenterCrop([input_size[0], input_size[1]]),
            # transforms.Resize([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
)

backbone = Backbone(input_size)
backbone.load_state_dict(torch.load('app/ms1mv3_arcface_r50_fp16/backbone_ir50_ms1m_epoch120.pth', map_location=torch.device("cpu")))
backbone.to(device)
backbone.eval()

# ****************************************************************************************************
# Create DL model to extract facial embedding
# print('\n**************************************************************************')

# print('\nLoading arcface model ...\n')
arcface_model = get_model('r50', fp16=False)
state_dict = torch.load('./app/arcface_torch/backbones/ms1mv3_arcface_r50_fp16/backbone.pth', map_location=device)
# state_dict = torch.load('app/ms1mv3_arcface_r50_fp16/backbone_ir50_ms1m_epoch120.pth', map_location=device)

arcface_model.load_state_dict(state_dict)
net.to(device)
arcface_model.eval()
# print('\nArcface model loaded\n')
# print('**************************************************************************\n')


# ****************************************************************************************************
model_dream = Branch(feat_dim=512)
# model.cuda()
checkpoint = torch.load('./app/arcface_torch/checkpoint_512.pth')
model_dream.load_state_dict(checkpoint['state_dict'])
model_dream.eval()

# 2. Introduce a pose estimator to solve pose.
# pose_estimator = PoseEstimator()

# 3. Introduce a mark detector to detect landmarks.
mark_detector = MarkDetector()



def cosine_similarity(x, y):
    return dot(x, y) / (sqrt(dot(x, x)) * sqrt(dot(y, y)))

def no_accent_vietnamese(utf8_str):
    INTAB = "ạảãàáâậầấẩẫăắằặẳẵóòọõỏôộổỗồốơờớợởỡéèẻẹẽêếềệểễúùụủũưựữửừứíìịỉĩýỳỷỵỹđẠẢÃÀÁÂẬẦẤẨẪĂẮẰẶẲẴÓÒỌÕỎÔỘỔỖỒỐƠỜỚỢỞỠÉÈẺẸẼÊẾỀỆỂỄÚÙỤỦŨƯỰỮỬỪỨÍÌỊỈĨÝỲỶỴỸĐ"
    OUTTAB = "a" * 17 + "o" * 17 + "e" * 11 + "u" * 11 + "i" * 5 + "y" * 5 + "d" + \
            "A" * 17 + "O" * 17 + "E" * 11 + "U" * 11 + "I" * 5 + "Y" * 5 + "D"
    r = re.compile("|".join(INTAB))
    replaces_dict = dict(zip(INTAB, OUTTAB))
    return r.sub(lambda m: replaces_dict[m.group(0)], utf8_str)

def loadBase64Img(uri):
    encoded_data = uri.split(',')[1]
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def load_image(img):
	exact_image = False; base64_img = False; url_img = False

	if type(img).__module__ == np.__name__:
		exact_image = True

	elif len(img) > 11 and img[0:11] == "data:image/":
		base64_img = True

	elif len(img) > 11 and img.startswith("http"):
		url_img = True

	#---------------------------

	if base64_img == True:
		img = loadBase64Img(img)

	elif url_img:
		img = np.array(Image.open(requests.get(img, stream=True).raw))

	elif exact_image != True: #image path passed as input
		if os.path.isfile(img) != True:
			raise ValueError("Confirm that ",img," exists")

		img = cv2.imread(img)

	return img


# def generate(id):
#     # cap = cv2.VideoCapture("rtsp://admin:Phunggi@911@" + ip_list.split('\n')[id] + ":554/profile2/media.smp")
#     while True:
#         ret, orig_image = cap.read()
#         if orig_image is None:
#             print("end")
#             break

#         image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
#         boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
#         boxes = boxes.detach().cpu().numpy()

#         gt_feats = []
#         for i in range(boxes.shape[0]):
#             box = boxes[i, :]
#             xmin, ymin, xmax, ymax = box
#             xmin -= (xmax-xmin)/18
#             xmax += (xmax-xmin)/18
#             ymin -= (ymax-ymin)/18
#             ymax += (ymax-ymin)/18
#             xmin = 0 if xmin < 0 else xmin
#             ymin = 0 if ymin < 0 else ymin
#             xmax = image.shape[1] if xmax >= image.shape[1] else xmax
#             ymax = image.shape[0] if ymax >= image.shape[0] else ymax
#             boxes[i,:] = [xmin, ymin, xmax, ymax]
#             infer_img = image[int(ymin): int(ymax), int(xmin): int(xmax), :]
#             if infer_img is not None and infer_img.shape[0] != 0 and infer_img.shape[1] != 0:
#                 with torch.no_grad():
#                     feat = F.normalize(backbone(transform(Image.fromarray(infer_img)).unsqueeze(0).to(device))).cpu()
#                 gt_feats.append(feat.detach().cpu().numpy())

#         for i in range(boxes.shape[0]):
#             box = boxes[i, :]
#             label ="un non"
#             if i < len(gt_feats):
#                 confs = np.dot(gt_feats[i], features.T)
#                 label = no_accent_vietnamese(names[np.argmax(confs)])
#                 label = f" {label}"
#             cv2.rectangle(orig_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 4)

#             cv2.putText(orig_image, label,
#                         (int(box[0]), int(box[1]) - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,  # font scale
#                         (0, 0, 255),
#                         2)  # line type
#         orig_image = cv2.resize(orig_image, None, None, fx=0.8, fy=0.8)
#         (flag, encodedImage) = cv2.imencode(".jpg", orig_image)
#         yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        
# @app.route("/video_feed/<id>")
# def video_feed(id):
# 	return Response(generate(int(id)), mimetype = "multipart/x-mixed-replace; boundary=frame")

@login_manager.unauthorized_handler
def unauthorized_handler():
    return redirect(url_for('index'))


@app.errorhandler(403)
def access_forbidden(error):
    return redirect(url_for('index'))


@app.errorhandler(404)
def not_found_error(error):
    return redirect(url_for('index'))


@app.errorhandler(500)
def internal_error(error):
    return redirect(url_for('index'))

@app.route('/', methods=["GET", "POST"])
def index():
    
    if 'login' in request.form and request.method == 'POST':

        # read form data
        username = request.form['username']
        password = request.form['password']

        # Locate user
        user = Users.query.filter_by(username=username).first()
        # Check the password
        if user and verify_pass(password, user.password):
            login_user(user)
        
        return redirect(url_for('index'))

    
    if 'register' in request.form and request.method == 'POST':

        username = request.form['username']
        password = request.form['password']

        if username.strip() == "" or password.strip() == "":
            return redirect(url_for('index'))

        # Check usename exists
        user = Users.query.filter_by(username=username).first()
        if user:
            return redirect(url_for('index'))
        # else we can create the user
        user = Users(username=username, password=password, secret_key="")
        db.session.add(user)
        db.session.commit()

        p = hnswlib.Index(space = 'cosine', dim = 512)
        p.init_index(max_elements = 1000, ef_construction = 200, M = 16)
        p.set_ef(10)
        p.set_num_threads(4)
        p.save_index("indexes/index_" + str(user.id) + ".bin")

        login_user(user)

        return redirect(url_for('index'))
    
    if not current_user.is_authenticated:
        return render_template('index.html', is_login=False)
    return render_template("index.html", is_login=True, current_user=current_user)
    
    
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route("/FaceRec", methods=['POST'])
def facerec():
    req = request.get_json()

    if 'secret_key' not in req:
        return jsonify({"result": {'message': 'Vui lòng truyền secret key'}}), 400

    user = Users.query.filter_by(secret_key=req['secret_key']).first()
    
    if not user:
        return jsonify({"result": {'message': 'Secret key không hợp lệ'}}), 403

    img_input = ""
    if "img" in list(req.keys()):
        img_input = req["img"]

    validate_img = False
    if len(img_input) > 11 and img_input[0:11] == "data:image/":
        validate_img = True

    if validate_img != True:
        return jsonify({"result": {'message': 'Vui lòng truyền ảnh dưới dạng Base64'}}), 400
    
    img = load_image(img_input)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
    # boxes = boxes.detach().cpu().numpy()
    boxes = np.array([[0, 0, image.shape[1], image.shape[0]]])

    feats = []
    images = []
    bboxes = []
    profile_face_ids = []
    generated_face_ids = []
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        xmin, ymin, xmax, ymax = box
        xmin -= (xmax-xmin)/18
        xmax += (xmax-xmin)/18
        ymin -= (ymax-ymin)/18
        ymax += (ymax-ymin)/18
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = image.shape[1] if xmax >= image.shape[1] else xmax
        ymax = image.shape[0] if ymax >= image.shape[0] else ymax
        boxes[i,:] = [xmin, ymin, xmax, ymax]
        infer_img = image[int(ymin): int(ymax), int(xmin): int(xmax), :]
        if infer_img is not None and infer_img.shape[0] != 0 and infer_img.shape[1] != 0:
            with torch.no_grad():
                feat = F.normalize(backbone(transform(Image.fromarray(infer_img)).unsqueeze(0).to(device))).cpu()
            feats.append(feat.detach().cpu().numpy())
            images.append(infer_img.copy())
            bboxes.append("{} {} {} {}".format(xmin, ymin, xmax, ymax))
            # profile_face_ids.append("")
            generated_face_ids.append("")
    
    p = hnswlib.Index(space = 'cosine', dim = 512)
    p.load_index("indexes/index_" + str(user.id) + '.bin')
    person_ids = []
    identities = []
    now = 0
    prenow = 0
    for feat, image in zip(feats, images):
        person_id = -1
        try:
            neighbors, distances = p.knn_query(feat, k=1)
            if distances[0][0] <= 0.5:
                person_id = db.session.query(DefineImages.person_id, func.count(DefineImages.person_id).label('total'))\
                            .filter(DefineImages.id.in_(neighbors.tolist()[0]))\
                            .filter(DefineImages.user_id==user.id)\
                            .group_by(DefineImages.person_id)\
                            .order_by(text('total DESC')).first().person_id
        except:
            person_id = -1
        
        person = People.query.filter_by(id=person_id).first()
        identities.append('Người lạ' if not person else person.name)
        person_ids.append(person_id)

        profile_image_id = DefineImages.query.filter_by(person_id=person_id).first()
        # profile_image = cv2.imread("images/" + req['secret_key'] + '/' + profile_image_id.image_id + ".jpg")
        profile_face_ids.append(profile_image_id.image_id if profile_image_id is not None else '')

        now = round(datetime.datetime.now().timestamp() * 1000)
        if prenow == now:
            now += 1
        if not os.path.isdir("images/" + req['secret_key'] ):
            os.mkdir("images/" + req['secret_key'] )
        cv2.imwrite("images/" + req['secret_key'] + "/face_" + str(now) + ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image = Timeline(user_id=user.id, person_id=person_id, image_id="face_" + str(now), embedding=np.array2string(feat, separator=','), timestamp=now)
        db.session.add(image)
        db.session.commit()
        prenow = now

        pub.sendMessage('face_vkist', uid=req['secret_key'], message='facerec ' + str(now))
    
    return jsonify({'result': {"bboxes": bboxes, "identities": identities, "id": person_ids, "profilefaceIDs": profile_face_ids, "3DFace": generated_face_ids}}), 200

@app.route("/FaceRec_DREAM", methods=['POST'])
def facerec_DREAM():

    req = request.get_json()

    if 'secret_key' not in req:
        return jsonify({"result": {'message': 'Vui lòng truyền secret key'}}), 400

    user = Users.query.filter_by(secret_key=req['secret_key']).first()
    
    if not user:
        return jsonify({"result": {'message': 'Secret key không hợp lệ'}}), 403

    img_input = ""
    if "img" in list(req.keys()):
        img_input = req["img"]

    validate_img = False
    if len(img_input) > 11 and img_input[0:11] == "data:image/":
        validate_img = True

    if validate_img != True:
        return jsonify({"result": {'message': 'Vui lòng truyền ảnh dưới dạng Base64'}}), 400
    
    img = load_image(img_input)

    # Step 1: Get a face from current frame.
    # raw_boxes, square_boxes = mark_detector.extract_cnn_facebox(img)
    # raw_boxes, square_boxes = mark_detector.extract_cnn_facebox(img)
    square_boxes = np.array([[0, 0, img.shape[1], img.shape[0]]])


    feats = []
    images = []
    bboxes = []
    profile_face_ids = []
    generated_face_ids = []
    # Any face found?
    # if facebox is not None:
    for bI, square_box_I in enumerate(square_boxes):
        # Step 2: Detect landmarks. Crop and feed the face area into the
        # mark detector.
        x1, y1, x2, y2 = square_box_I
        face_img_I = img[y1: y2, x1: x2]

        # Run the detection.
        marks = mark_detector.detect_marks(face_img_I)
        # print('Landmark detection in {:.2f} s'.format(tm.getTimeSec()))
        # Convert the locations from local face area to the global image.
        marks *= (x2 - x1)
        marks[:, 0] += x1
        marks[:, 1] += y1
        
        # POSE ESTIMATION

        imgpts, modelpts, rotate_degree, nose, landmark_6p = face_orientation(img, marks)

        # EXTRACT EMBEDDING

        x3, y3, x4, y4 = square_boxes[bI]
        # if x3 < 0 or y3 < 0 or x4 > frame_width or y4 >frame_height:
        #     continue
        # cropped_face_I = cv2.resize(img[y3: y4, x3: x4], (112, 112))
        # cropped_face_I = cv2.cvtColor(cropped_face_I, cv2.COLOR_BGR2RGB)
        # cropped_face_I = np.transpose(cropped_face_I, (2, 0, 1))
        # cropped_face_I = torch.from_numpy(cropped_face_I).unsqueeze(0).float()
        # cropped_face_I.div_(255).sub_(0.5).div_(0.5)

        # embedding_I = arcface_model(cropped_face_I.to(device)).detach().numpy()[0]
        with torch.no_grad():
            embedding_I = F.normalize(backbone(transform(Image.fromarray(img[y3: y4, x3: x4])).unsqueeze(0).to(device))).cpu()
        # DREAM
        # print('Adding residual to current embedding')
        yaw = np.zeros([1, 1])
        yaw[0,0] = norm_angle(float(rotate_degree[2]))
        original_embedding_tensor = np.expand_dims(embedding_I.detach().cpu().numpy(), axis=0)
        # feat = torch.autograd.Variable(torch.from_numpy(feat.astype(np.float32)), volatile=True).cuda()
        # yaw = torch.autograd.Variable(torch.from_numpy(yaw.astype(np.float32)), volatile=True).cuda()
        feature_original = torch.autograd.Variable(torch.from_numpy(original_embedding_tensor.astype(np.float32)))
        yaw = torch.autograd.Variable(torch.from_numpy(yaw.astype(np.float32)))

        new_embedding = model_dream(feature_original, yaw)
        # new_embedding = new_embedding.cpu().data.numpy()
        new_embedding = new_embedding.to(device).data.numpy()
        embedding_I = new_embedding[0, :]


        feats.append(embedding_I)
        images.append(cv2.cvtColor(img[y3: y4, x3: x4], cv2.COLOR_BGR2RGB).copy())
        bboxes.append("{} {} {} {}".format(x3, y3, x4, y4))
        # profile_face_ids.append("")
        generated_face_ids.append("")
 
    p = hnswlib.Index(space = 'cosine', dim = 512)
    p.load_index("indexes/index_" + str(user.id) + '.bin')
    person_ids = []
    identities = []
    now = 0
    prenow = 0
    for feat, image in zip(feats, images):
        person_id = -1
        try:
            neighbors, distances = p.knn_query(feat, k=1)
            if distances[0][0] <= 0.5:
                person_id = db.session.query(DefineImages.person_id, func.count(DefineImages.person_id).label('total'))\
                            .filter(DefineImages.id.in_(neighbors.tolist()[0]))\
                            .filter(DefineImages.user_id==user.id)\
                            .group_by(DefineImages.person_id)\
                            .order_by(text('total DESC')).first().person_id
        except:
            person_id = -1
        
        person = People.query.filter_by(id=person_id).first()
        identities.append('Người lạ' if not person else person.name)
        person_ids.append(person_id)

        profile_image_id = DefineImages.query.filter_by(person_id=person_id).first()
        # profile_image = cv2.imread("images/" + req['secret_key'] + '/' + profile_image_id.image_id + ".jpg")
        profile_face_ids.append(profile_image_id.image_id if profile_image_id is not None else '')

        now = round(datetime.datetime.now().timestamp() * 1000)
        if prenow == now:
            now += 1
        if not os.path.isdir("images/" + req['secret_key'] ):
            os.mkdir("images/" + req['secret_key'] )
        cv2.imwrite("images/" + req['secret_key'] + "/face_" + str(now) + ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image = Timeline(user_id=user.id, person_id=person_id, image_id="face_" + str(now), embedding=np.array2string(feat, separator=','), timestamp=now)
        db.session.add(image)
        db.session.commit()
        prenow = now

        pub.sendMessage('face_vkist', uid=req['secret_key'], message='facerec ' + str(now))
    
    return jsonify({'result': {"bboxes": bboxes, "identities": identities, "id": person_ids, "profilefaceIDs": profile_face_ids, "3DFace": generated_face_ids}}), 200

@app.route("/FaceRec_3DFaceModeling", methods=['POST'])
def facerec_3DFaceModeling():
    req = request.get_json()

    if 'secret_key' not in req:
        return jsonify({"result": {'message': 'Vui lòng truyền secret key'}}), 400

    user = Users.query.filter_by(secret_key=req['secret_key']).first()
    
    if not user:
        return jsonify({"result": {'message': 'Secret key không hợp lệ'}}), 403

    img_input = ""
    if "img" in list(req.keys()):
        img_input = req["img"]

    validate_img = False
    if len(img_input) > 11 and img_input[0:11] == "data:image/":
        validate_img = True

    if validate_img != True:
        return jsonify({"result": {'message': 'Vui lòng truyền ảnh dưới dạng Base64'}}), 400
    
    api = 'http://10.1.12.201:5000/3dface'

    # import requests
    # import cv2
    # import base64

    # for i in range(0,10):
    #     image = cv2.imread("C:\\Users\\X270\\Downloads\\images\\" + str(i+1).zfill(3) + ".jpg")

    #     retval, buffer = cv2.imencode('.jpg', image)
    #     jpg_as_text = base64.b64encode(buffer)

    #     res = requests.post("http://123.16.55.212:85/facerec", json={"img": "data:image/jpeg;base64," + str(jpg_as_text)[2:-1]})
    #     print(res.json())
    response = requests.post(api, json={"img": img_input}, timeout=100)
    # print(response.json()['result'])
    # Request frontalization face image from 3D face modeling
    frontal_3d_face = response.json()['result']
    # frontal_3d_face = img_input
    img = load_image(frontal_3d_face)


    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
    # boxes = boxes.detach().cpu().numpy()

    boxes = np.array([[0, 0, image.shape[1], image.shape[0]]])

    feats = []
    images = []
    bboxes = []
    profile_face_ids = []
    generated_face_ids = []
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        xmin, ymin, xmax, ymax = box
        xmin -= (xmax-xmin)/18
        xmax += (xmax-xmin)/18
        ymin -= (ymax-ymin)/18
        ymax += (ymax-ymin)/18
        xmin = 0 if xmin < 0 else xmin
        ymin = 0 if ymin < 0 else ymin
        xmax = image.shape[1] if xmax >= image.shape[1] else xmax
        ymax = image.shape[0] if ymax >= image.shape[0] else ymax
        boxes[i,:] = [xmin, ymin, xmax, ymax]
        infer_img = image[int(ymin): int(ymax), int(xmin): int(xmax), :]
        if infer_img is not None and infer_img.shape[0] != 0 and infer_img.shape[1] != 0:
            with torch.no_grad():
                feat = F.normalize(backbone(transform(Image.fromarray(infer_img)).unsqueeze(0).to(device))).cpu()
            feats.append(feat.detach().cpu().numpy())
            images.append(infer_img.copy())
            bboxes.append("{} {} {} {}".format(xmin, ymin, xmax, ymax))
            # profile_face_ids.append("")
            # generated_faces.append(frontal_3d_face)
    
    p = hnswlib.Index(space = 'cosine', dim = 512)
    p.load_index("indexes/index_" + str(user.id) + '.bin')
    person_ids = []
    identities = []
    now = 0
    prenow = 0
    for feat, image in zip(feats, images):
        person_id = -1
        try:
            neighbors, distances = p.knn_query(feat, k=1)
            if distances[0][0] <= 0.5:
                person_id = db.session.query(DefineImages.person_id, func.count(DefineImages.person_id).label('total'))\
                            .filter(DefineImages.id.in_(neighbors.tolist()[0]))\
                            .filter(DefineImages.user_id==user.id)\
                            .group_by(DefineImages.person_id)\
                            .order_by(text('total DESC')).first().person_id
        except:
            person_id = -1
        
        person = People.query.filter_by(id=person_id).first()
        identities.append('Người lạ' if not person else person.name)
        person_ids.append(person_id)
        profile_image_id = DefineImages.query.filter_by(person_id=person_id).first()
        # profile_image = cv2.imread("images/" + req['secret_key'] + '/' + profile_image_id.image_id + ".jpg")
        profile_face_ids.append(profile_image_id.image_id if profile_image_id is not None else '')

        now = round(datetime.datetime.now().timestamp() * 1000)
        if prenow == now:
            now += 1
        if not os.path.isdir("images/" + req['secret_key'] ):
            os.mkdir("images/" + req['secret_key'] )
        cv2.imwrite("images/" + req['secret_key'] + "/face_" + str(now) + ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        image = Timeline(user_id=user.id, person_id=person_id, image_id="face_" + str(now), embedding=np.array2string(feat, separator=','), timestamp=now)
        db.session.add(image)
        db.session.commit()
        prenow = now
        generated_face_ids.append("face_" + str(now))
        pub.sendMessage('face_vkist', uid=req['secret_key'], message='facerec ' + str(now))
    
    return jsonify({'result': {"bboxes": bboxes, "identities": identities, "id": person_ids, "profilefaceIDs": profile_face_ids, "3DFace": generated_face_ids}}), 200

@app.route('/facereg', methods=['POST'])
@login_required
def facereg():
    req = request.get_json()
    if 'image_id' not in req or not ('name' in req or 'access_key' in req):
        return jsonify({"result": {'message': 'Vui lòng truyền id của ảnh và id đối tượng'}}), 400
    
    image_id = req['image_id']
    
    if not ('access_key' in req):
        name = req['name']
        access_key = str(uuid.uuid4())
    else:
        access_key = req['access_key']

    p = hnswlib.Index(space = 'cosine', dim = 512)
    p.load_index("indexes/index_" + str(current_user.id) + '.bin', max_elements=1000)

    image_instance = Timeline.query.filter_by(image_id=image_id).first()
    embedding = image_instance.embedding
    embedding = embedding[2:-2]
    embedding = np.expand_dims(np.fromstring(embedding, dtype='float32', sep=','), axis=0)

    if not ('access_key' in req):
        person = People(user_id=current_user.id, name=name, access_key=access_key)
        db.session.add(person)
        db.session.commit()

    person = People.query.filter_by(access_key=access_key).first()
    define_image = DefineImages(user_id=current_user.id, person_id=person.id, image_id=image_id)
    db.session.add(define_image)
    db.session.commit()

    image_instance = Timeline.query.filter_by(image_id=image_id).first()
    image_instance.person_id = person.id
    db.session.commit()

    define_image = DefineImages.query.filter_by(image_id=image_id).first()
    p.add_items(embedding, np.array([define_image.id]))
    p.save_index("indexes/index_" + str(current_user.id) + '.bin')

    now = datetime.datetime.now().timestamp() * 1000
    pub.sendMessage('face_vkist', uid=current_user.secret_key, message='facerec ' + str(now))

    return jsonify({"result": {'message': 'success'}}), 200


@app.route("/data")
@login_required
def data():
    today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000
    
    all_people = db.session.query(DefineImages.person_id, DefineImages.image_id, People.name, People.access_key)\
              .filter(DefineImages.user_id==current_user.id)\
              .filter(People.id==DefineImages.person_id)\
              .group_by(DefineImages.person_id, People.name, People.access_key)\
              .all()

    current_checkin = db.session.query(Timeline.person_id, Timeline.timestamp, Timeline.image_id, People.name)\
              .filter(Timeline.user_id==current_user.id)\
              .filter(People.id==Timeline.person_id)\
              .filter(Timeline.timestamp >= today)\
              .group_by(Timeline.person_id, People.name)\
              .all()

    current_timeline = db.session.query(Timeline.person_id, Timeline.timestamp, Timeline.image_id, People.name)\
              .filter(Timeline.user_id==current_user.id)\
              .filter(People.id==Timeline.person_id)\
              .order_by(Timeline.timestamp.desc())\
              .limit(10)\
              .all()

    strangers = db.session.query(Timeline.person_id, Timeline.timestamp, Timeline.image_id)\
              .filter(Timeline.user_id==current_user.id)\
              .filter(Timeline.person_id==-1)\
              .order_by(Timeline.timestamp.desc())\
              .limit(10)\
              .all()

    if not current_checkin:
        current_checkin = []
    people_array = {}
    for u in all_people:
        people_array[str(u[0])] = {'name': u[2], 'image_id': u[1], 'timestamp': '--', 'checkin': False, 'access_key': u[3]}
    for u in current_checkin:
        people_array[str(u[0])] = {'name': u[3], 'image_id': u[2], 'timestamp': str(u[1]), 'checkin': True, 'access_key': people_array[str(u[0])]['access_key']}
    
    number_of_current_checkin = len(current_checkin)
    current_checkin = [people_array[u] for u in people_array.keys()]
    current_timeline = [{'name': u[3], 'image_id': u[2], 'timestamp': str(u[1])} for u in current_timeline]
    strangers = [{'image_id': u[2], 'timestamp': str(u[1])} for u in strangers]
    number_of_people = len(current_checkin)

    t = 0
    r = 0
    a = 0
    if torch.cuda.is_available():
        t = torch.cuda.get_device_properties(0).total_memory
        r = torch.cuda.memory_reserved(0)
        a = torch.cuda.memory_allocated(0)

    return jsonify({
        "result": {
            'secret_key': current_user.secret_key,
            'number_of_people': number_of_people,
            'current_checkin': current_checkin,
            'number_of_current_checkin': number_of_current_checkin,
            'current_timeline': current_timeline,
            'strangers': strangers,
            'gpu': {
                't': t,
                'r': r,
                'a': a
            }
        }
    }), 200

@app.route("/images/<secret_key>/<image_id>")
def images(secret_key, image_id):
    return send_file('images/' + secret_key + "/" + image_id + '.jpg')

socketio.run(app, host='0.0.0.0', port=5051)
# app.run(host='0.0.0.0', port=5000)
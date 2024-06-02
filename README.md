from flask import Flask, request, jsonify
import cv2
import dlib
import numpy as np
from PIL import Image
import json

app = Flask(__name__)

# Initialize the face detector
face_detector = dlib.get_frontal_face_detector()

# Sample tenant and org lists
Tenant_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Arg_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

def detect_faces(image):
    faces = face_detector(image, 1)
    faces_info = []
    
    for face_count, face in enumerate(faces, start=1):
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y

        faces_info.append({'face_id': face_count, 'coordinates': {'x': x, 'y': y, 'width': w, 'height': h}})
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imwrite("output_detection.jpg", image)
    print("Number of faces with rectangles drawn:", len(faces_info))
    return faces_info

@app.route('/detect_faces', methods=["POST"])
def detect_faces_api():
    try:
        tenant_id = request.form.get('Tenant_Id')
        org_id = request.form.get('Org_Id')

        if tenant_id is None or org_id is None:
            return jsonify({'error': 'Tenant_Id and Org_id are required', 'status': 0}), 400

        if int(tenant_id) not in Tenant_list or int(org_id) not in Arg_list:
            return jsonify({'error': 'Invalid Tenant_Id or Org_id', 'status': 0}), 400

        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided', 'status': 0}), 400

        file = request.files['image']
        image = Image.open(file.stream)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        faces_info = detect_faces(image)
        
        if not faces_info:
            return jsonify({'error': 'Face not Detected', 'status': 0}), 400

        return jsonify({'status': 1, 'faces_info': faces_info})

    except Exception as e:
        print(f"Exception: {str(e)}")
        return jsonify({'error': 'Internal Server Error', 'message': str(e), 'status': 0}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

import logging
import time
from flask import Flask
from flask import request
from flask_cors import cross_origin

from util.logger import create_logger
from util.response import json_response, error
from src import face_match_controller
import os
from settings import IMG_PATH

app = Flask(__name__)

logger = create_logger('app', logging.DEBUG, 'app.log')


@app.route('/facematch/compare', methods=['POST'])
@cross_origin()
def compare_face():
    if 'files' not in request.files:
        return json_response(error(400, "no files sent"), None)
    uploaded_files = request.files.getlist("files")

    faces = []
    for f in uploaded_files:
        file_path = os.path.join(IMG_PATH, f.filename)
        faces.append(file_path)
        f.save(file_path)

    try:
        start_time = time.time()
        dist = face_match_controller.compare_two_faces(faces[0], faces[1])
        result = 1 if dist < 1 else 0

        return json_response(None, {
            "distance": dist,
            "valid": result,
            "time_taken": time.time() - start_time
        })
    except Exception as e:
        logger.exception(e)
        return json_response(error(400, "cannot find face in one or both image"), None)


@app.route('/facematch/health_check', methods=['GET'])
@cross_origin()
def health_check():
    return "ok"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
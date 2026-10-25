
import cv2
import numpy as np
from insightface.app import FaceAnalysis

_app = None
def get_face_app():
    global _app
    if _app is None:
        _app = FaceAnalysis(name="buffalo_l")
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app

def extract_embedding(image_bgr):
    app = get_face_app()
    faces = app.get(image_bgr)
    if len(faces) == 0:
        return None, None
    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
    face = faces[0]
    return face.normed_embedding, face.bbox

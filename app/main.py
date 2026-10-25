
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os, shutil, uuid
from .face_store import FaceVectorStore
from .face_engine import extract_embedding

app = FastAPI(title="Face RAG FastAPI", version="1.0")

STORE = FaceVectorStore(dim=512, store_path="./face_store")
UPLOAD_DIR = "./face_store/images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app.mount("/images", StaticFiles(directory=UPLOAD_DIR), name="images")

camera = None

def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError("USB camera open fail")
    return camera

@app.get("/")
def root():
    return {"msg": "Face RAG Server", "total_faces": STORE.index.ntotal}

@app.post("/register")
async def register_face(label: str = Form(...), file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    fname = f"{label}_{uuid.uuid4().hex[:8]}{ext}"
    fpath = os.path.join(UPLOAD_DIR, fname)
    with open(fpath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    img = cv2.imread(fpath)
    if img is None:
        raise HTTPException(400, "image read fail")
    emb, bbox = extract_embedding(img)
    if emb is None:
        os.remove(fpath)
        raise HTTPException(400, "face not found")
    STORE.add(emb, label, fpath)
    STORE.save_embedding(emb)
    return {"status": "ok", "label": label, "bbox": bbox.tolist() if bbox is not None else None, "total": STORE.index.ntotal}

@app.get("/capture/recognize")
def capture_and_recognize(k: int = 3, threshold: float = 0.35):
    cap = get_camera()
    ret, frame = cap.read()
    if not ret:
        raise HTTPException(500, "camera capture fail")
    emb, bbox = extract_embedding(frame)
    if emb is None:
        return {"status": "no_face", "message": "no face detected"}
    results = STORE.search(emb, k=k, threshold=threshold)
    if not results:
        return {"status": "unknown", "bbox": bbox.tolist()}
    return {"status": "found", "bbox": bbox.tolist(), "matches": results, "best": results[0]}

@app.post("/recognize/upload")
async def recognize_upload(file: UploadFile = File(...), k: int = 3, threshold: float = 0.35):
    data = await file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    emb, bbox = extract_embedding(img)
    if emb is None:
        return {"status": "no_face"}
    results = STORE.search(emb, k=k, threshold=threshold)
    return {"status": "found" if results else "unknown", "matches": results, "bbox": bbox.tolist() if bbox is not None else None}

@app.get("/labels")
def list_labels():
    counter, metas = STORE.list_labels()
    return {"total_vectors": STORE.index.ntotal, "counts": dict(counter), "metas": metas}

@app.delete("/labels/{label}")
def delete_label(label: str):
    ok = STORE.delete_by_label(label)
    if not ok:
        raise HTTPException(500, "delete fail")
    return {"status": "deleted", "label": label, "remaining": STORE.index.ntotal}

@app.post("/camera/release")
def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return {"status": "released"}

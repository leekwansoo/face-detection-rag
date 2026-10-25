
import streamlit as st
import cv2, os, shutil, uuid, numpy as np, pickle, faiss, sys
from PIL import Image
st.set_page_config(page_title="Face RAG - InsightFace", layout="wide")

# --- Backend (동일 로직을 UI 안에 포함) ---
from insightface.app import FaceAnalysis

@st.cache_resource
def get_face_app():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(640,640))
    return app

@st.cache_resource
def get_store():
    class Store:
        def __init__(self):
            self.store_path="./face_store"
            os.makedirs(self.store_path, exist_ok=True)
            os.makedirs(f"{self.store_path}/images", exist_ok=True)
            self.index_path=f"{self.store_path}/faiss.index"
            self.meta_path=f"{self.store_path}/meta.pkl"
            self.emb_path=f"{self.store_path}/embeddings.npy"
            if os.path.exists(self.index_path):
                self.index=faiss.read_index(self.index_path)
            else:
                self.index=faiss.IndexFlatIP(512)
            self.metas=pickle.load(open(self.meta_path,"rb")) if os.path.exists(self.meta_path) else []
        def add(self, emb, label, img_path):
            if emb.ndim==1: emb=emb[None,:]
            faiss.normalize_L2(emb)
            self.index.add(emb)
            self.metas.append({"label":label,"image_path":img_path})
            faiss.write_index(self.index,self.index_path)
            pickle.dump(self.metas, open(self.meta_path,"wb"))
            np.save(self.emb_path, np.vstack([np.load(self.emb_path), emb]) if os.path.exists(self.emb_path) else emb)
        def search(self, emb, k=3, thr=0.35):
            if self.index.ntotal==0: return []
            if emb.ndim==1: emb=emb[None,:]
            faiss.normalize_L2(emb)
            scores, idxs=self.index.search(emb,k)
            res=[]
            for sc, idx in zip(scores[0], idxs[0]):
                if idx==-1 or sc<thr: continue
                res.append({"label":self.metas[idx]["label"],"score":float(sc),"idx":int(idx)})
            return sorted(res,key=lambda x:x["score"],reverse=True)
    return Store()

def extract_emb(image_bgr):
    app=get_face_app()
    faces=app.get(image_bgr)
    if not faces: return None, None
    faces=sorted(faces, key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
    return faces[0].normed_embedding, faces[0].bbox

st.title("🔍 Face RAG - InsightFace (512dim) UI")
st.caption("고성능 buffalo_l 모델 | Cosine threshold 0.35~0.6 추천")

store=get_store()
col1, col2 = st.columns([1,1])

with col1:
    st.subheader("1️⃣ 얼굴 등록")
    label = st.text_input("라벨 (이름)", placeholder="Kwansoo")
    files = st.file_uploader("얼굴 사진 (여러장 가능)", type=["jpg","png"], accept_multiple_files=True)
    if st.button("등록하기", type="primary", disabled=not (label and files)):
        for f in files:
            fname=f"{label}_{uuid.uuid4().hex[:6]}.jpg"
            fpath=os.path.join(store.store_path,"images",fname)
            with open(fpath,"wb") as out:
                out.write(f.getbuffer())
            img=cv2.imread(fpath)
            emb,bbox=extract_emb(img)
            if emb is None:
                os.remove(fpath)
                st.warning(f"{f.name}: 얼굴 못찾음")
            else:
                store.add(emb,label,fpath)
                st.success(f"{f.name} -> {label} 등록 완료")
        st.rerun()

    st.divider()
    st.subheader("📋 등록 현황")
    from collections import Counter
    c=Counter([m["label"] for m in store.metas])
    st.metric("총 벡터", store.index.ntotal)
    st.json(dict(c))
    if st.button("전체 삭제"):
        if os.path.exists(store.index_path): os.remove(store.index_path)
        if os.path.exists(store.meta_path): os.remove(store.meta_path)
        if os.path.exists(store.emb_path): os.remove(store.emb_path)
        st.rerun()

with col2:
    st.subheader("2️⃣ USB 카메라 실시간 인식")
    thr = st.slider("Threshold (낮을수록 관대)", 0.2, 0.8, 0.35, 0.05)
    k = st.slider("Top-K", 1, 5, 3)
    
    run_cam = st.checkbox("카메라 켜기", value=False)
    FRAME_WINDOW = st.image([])
    
    if run_cam:
        cap=cv2.VideoCapture(0)
        st.info("카메라 실행 중... q 누르거나 체크 해제하면 종료")
        while run_cam:
            ret, frame = cap.read()
            if not ret: break
            emb,bbox=extract_emb(frame)
            if emb is not None:
                res=store.search(emb,k=k,thr=thr)
                x1,y1,x2,y2 = map(int,bbox)
                label_text = f"{res[0]['label']} {res[0]['score']:.2f}" if res else "Unknown"
                color = (0,255,0) if res else (0,0,255)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                cv2.putText(frame,label_text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                if res:
                    st.toast(f"Found: {res[0]['label']} ({res[0]['score']:.2f})")
            FRAME_WINDOW.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
            run_cam = st.session_state.get("camera", True)
            if not st.checkbox("카메라 유지", value=True, key="keep"):
                break
        cap.release()

    st.divider()
    st.subheader("📤 이미지 업로드 인식 테스트")
    test_file = st.file_uploader("테스트 이미지", type=["jpg","png"], key="test")
    if test_file:
        img = Image.open(test_file)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        emb,bbox=extract_emb(img_bgr)
        if emb is None:
            st.error("얼굴 없음")
        else:
            res=store.search(emb,k=k,thr=thr)
            st.json(res if res else {"status":"unknown"})
            if bbox is not None:
                x1,y1,x2,y2=map(int,bbox)
                cv2.rectangle(img_bgr,(x1,y1),(x2,y2),(0,255,0),2)
            st.image(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB))

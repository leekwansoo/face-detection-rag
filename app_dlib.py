
import streamlit as st
import cv2, os, uuid, numpy as np, pickle, faiss
from PIL import Image
import face_recognition
st.set_page_config(page_title="Face RAG - dlib", layout="wide")

@st.cache_resource
def get_store():
    class Store:
        def __init__(self):
            self.store_path="./face_store_dlib"
            os.makedirs(self.store_path, exist_ok=True)
            os.makedirs(f"{self.store_path}/images", exist_ok=True)
            self.index_path=f"{self.store_path}/faiss.index"
            self.meta_path=f"{self.store_path}/meta.pkl"
            self.emb_path=f"{self.store_path}/embeddings.npy"
            if os.path.exists(self.index_path):
                self.index=faiss.read_index(self.index_path)
            else:
                self.index=faiss.IndexFlatIP(128)
            self.metas=pickle.load(open(self.meta_path,"rb")) if os.path.exists(self.meta_path) else []
        def add(self, emb, label, img_path):
            if emb.ndim==1: emb=emb[None,:]
            faiss.normalize_L2(emb)
            self.index.add(emb)
            self.metas.append({"label":label,"image_path":img_path})
            faiss.write_index(self.index,self.index_path)
            pickle.dump(self.metas, open(self.meta_path,"wb"))
            np.save(self.emb_path, np.vstack([np.load(self.emb_path), emb]) if os.path.exists(self.emb_path) else emb)
        def search(self, emb, k=3, thr=0.45):
            if self.index.ntotal==0: return []
            if emb.ndim==1: emb=emb[None,:]
            faiss.normalize_L2(emb)
            scores, idxs=self.index.search(emb,k)
            res=[]
            for sc, idx in zip(scores[0], idxs[0]):
                if idx==-1 or sc<thr: continue
                res.append({"label":self.metas[idx]["label"],"score":float(sc),"distance":float(1-sc),"idx":int(idx)})
            return sorted(res,key=lambda x:x["score"],reverse=True)
    return Store()

def extract_emb(image_bgr):
    rgb=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    locs=face_recognition.face_locations(rgb, model="hog")
    if not locs: return None, None
    encs=face_recognition.face_encodings(rgb, locs)
    if not encs: return None, None
    return encs[0], locs[0]

def extract_multi(image_bgr):
    rgb=cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)
    locs=face_recognition.face_locations(rgb)
    encs=face_recognition.face_encodings(rgb, locs)
    return list(zip(encs, locs))

st.title("🙂 Face RAG - dlib (128dim) UI")
st.caption("가볍고 빠른 hog 모델 | Threshold 0.45~0.6 추천 | 라즈베리파이/저사양 추천")

store=get_store()
col1,col2 = st.columns([1,1])

with col1:
    st.subheader("1️⃣ 얼굴 등록")
    label = st.text_input("라벨 (이름)", placeholder="Kwansoo", key="d_label")
    files = st.file_uploader("얼굴 사진 (여러장)", type=["jpg","png"], accept_multiple_files=True, key="d_files")
    if st.button("등록하기", type="primary", disabled=not (label and files), key="d_reg"):
        for f in files:
            fname=f"{label}_{uuid.uuid4().hex[:6]}.jpg"
            fpath=os.path.join(store.store_path,"images",fname)
            with open(fpath,"wb") as out:
                out.write(f.getbuffer())
            img=cv2.imread(fpath)
            emb,loc=extract_emb(img)
            if emb is None:
                os.remove(fpath)
                st.warning(f"{f.name}: 얼굴 못찾음")
            else:
                store.add(emb,label,fpath)
                st.success(f"{f.name} -> {label} 등록")
        st.rerun()

    st.divider()
    st.subheader("📋 등록 현황")
    from collections import Counter
    c=Counter([m["label"] for m in store.metas])
    col_a,col_b = st.columns(2)
    col_a.metric("총 벡터", store.index.ntotal)
    col_b.metric("라벨 수", len(c))
    st.json(dict(c))
    # 이미지 그리드
    if store.metas:
        st.write("최근 등록 이미지")
        cols=st.columns(4)
        for i,m in enumerate(store.metas[-8:]):
            if os.path.exists(m["image_path"]):
                cols[i%4].image(m["image_path"], caption=m["label"], use_container_width=True)

with col2:
    st.subheader("2️⃣ USB 카메라 실시간 인식")
    thr = st.slider("Threshold", 0.2, 0.8, 0.45, 0.05, key="d_thr")
    k = st.slider("Top-K", 1, 5, 1, key="d_k")
    multi = st.checkbox("여러명 동시 인식 모드", value=False)

    cam_on = st.checkbox("카메라 켜기", value=False, key="d_cam")
    FRAME = st.image([])

    if cam_on:
        cap=cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("카메라를 열 수 없습니다. 0번이 아니면 1,2로 바꿔보세요")
        else:
            placeholder = st.empty()
            while True:
                ret, frame = cap.read()
                if not ret: break
                if multi:
                    items=extract_multi(frame)
                    for emb, (top,right,bottom,left) in items:
                        res=store.search(emb,k=1,thr=thr)
                        txt = f"{res[0]['label']} {res[0]['score']:.2f}" if res else "Unknown"
                        col=(0,255,0) if res else (0,0,255)
                        cv2.rectangle(frame,(left,top),(right,bottom),col,2)
                        cv2.putText(frame,txt,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,col,2)
                else:
                    emb, loc = extract_emb(frame)
                    if emb is not None:
                        top,right,bottom,left = loc
                        res=store.search(emb,k=k,thr=thr)
                        txt = f"{res[0]['label']} {res[0]['score']:.2f}" if res else "Unknown"
                        col=(0,255,0) if res else (0,0,255)
                        cv2.rectangle(frame,(left,top),(right,bottom),col,2)
                        cv2.putText(frame,txt,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,col,2)
                        if res:
                            placeholder.success(f"✅ {res[0]['label']} - 유사도 {res[0]['score']:.3f}")
                FRAME.image(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
                # Streamlit 루프 탈출용 - 다시 체크
                if not st.session_state.get("d_cam", False):
                    break
            cap.release()

    st.divider()
    st.subheader("📤 업로드 테스트")
    test = st.file_uploader("테스트 이미지", type=["jpg","png"], key="d_test")
    if test:
        img = Image.open(test)
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        emb, loc = extract_emb(img_bgr)
        if emb is None:
            st.error("얼굴 없음")
        else:
            res=store.search(emb,k=k,thr=thr)
            st.write("결과:", res if res else "Unknown")
            if loc:
                top,right,bottom,left=loc
                cv2.rectangle(img_bgr,(left,top),(right,bottom),(0,255,0),2)
            st.image(cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB))

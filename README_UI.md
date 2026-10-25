
# Streamlit UI - 두 버전

## 1. InsightFace 고성능 버전 (512 dim)
```bash
pip install insightface faiss-cpu opencv-python streamlit
streamlit run app_insightface.py --server.port 8501
```

## 2. dlib 가벼운 버전 (128 dim)
```bash
pip install face-recognition faiss-cpu opencv-python streamlit
streamlit run app_dlib.py --server.port 8502
```

둘 다 동시에 띄워서 비교 가능:
- 8501: InsightFace
- 8502: dlib

## 기능
- 왼쪽: 라벨 입력 + 사진 여러장 드래그로 등록
- 오른쪽: USB 카메라 실시간 박스 + 라벨 표시
- threshold 슬라이더로 엄격도 조절
- 업로드 테스트로 사진으로 인식 확인

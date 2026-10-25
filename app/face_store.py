
import os
import pickle
import numpy as np
import faiss

class FaceVectorStore:
    def __init__(self, dim=512, store_path="./face_store"):
        self.dim = dim
        self.store_path = store_path
        os.makedirs(store_path, exist_ok=True)
        self.index_path = os.path.join(store_path, "faiss.index")
        self.meta_path = os.path.join(store_path, "meta.pkl")
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = faiss.IndexFlatIP(dim)
        if os.path.exists(self.meta_path):
            with open(self.meta_path, "rb") as f:
                self.metas = pickle.load(f)
        else:
            self.metas = []

    def add(self, embedding, label, image_path=""):
        import numpy as np
        if embedding.ndim == 1:
            embedding = embedding[None, :]
        faiss.normalize_L2(embedding)
        self.index.add(embedding)
        self.metas.append({"label": label, "image_path": image_path})
        self.save()

    def search(self, embedding, k=3, threshold=0.35):
        if self.index.ntotal == 0:
            return []
        if embedding.ndim == 1:
            embedding = embedding[None, :]
        faiss.normalize_L2(embedding)
        scores, idxs = self.index.search(embedding, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            if score < threshold:
                continue
            meta = self.metas[idx]
            results.append({"label": meta["label"], "score": float(score), "index": int(idx)})
        return sorted(results, key=lambda x: x["score"], reverse=True)

    def list_labels(self):
        from collections import Counter
        return Counter([m["label"] for m in self.metas]), self.metas

    def delete_by_label(self, label):
        emb_path = os.path.join(self.store_path, "embeddings.npy")
        if not os.path.exists(emb_path):
            return False
        all_embs = np.load(emb_path)
        new_embs = []
        keep_metas = []
        for i, m in enumerate(self.metas):
            if m["label"] != label:
                new_embs.append(all_embs[i])
                keep_metas.append(m)
        if len(new_embs) == 0:
            self.index.reset()
            self.metas = []
            if os.path.exists(emb_path):
                os.remove(emb_path)
        else:
            new_embs = np.vstack(new_embs)
            faiss.normalize_L2(new_embs)
            self.index.reset()
            self.index.add(new_embs)
            self.metas = keep_metas
            np.save(emb_path, new_embs)
        self.save()
        return True

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metas, f)

    def save_embedding(self, emb):
        emb_path = os.path.join(self.store_path, "embeddings.npy")
        if emb.ndim == 1:
            emb = emb[None, :]
        if os.path.exists(emb_path):
            existing = np.load(emb_path)
            all_embs = np.vstack([existing, emb])
        else:
            all_embs = emb
        np.save(emb_path, all_embs)

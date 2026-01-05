import streamlit as st
import pandas as pd
import numpy as np
import pickle
import faiss
import re
import os
import shutil
import zipfile
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download

# ============================================
# 1. CẤU HÌNH & CSS
# ============================================
st.set_page_config(page_title="H&M AI Shop", page_icon="🛍️", layout="wide")

st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {width: 100%; border-radius: 5px; font-weight: bold;}
    .block-container {padding-top: 2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.1rem;}
</style>
""", unsafe_allow_html=True)

# ============================================
# 2. HỆ THỐNG TẢI ẢNH TỪ DATASET (CACHE)
# ============================================
# 👉 SỬA LẠI THÔNG TIN NÀY CHO ĐÚNG CỦA BRO
DATASET_REPO_ID = "stephenhoang/hm-fashion-images-demo" 
ZIP_FILENAME = "hm_images_50k_optimized.zip" # Tên file zip bro đã up lên dataset
LOCAL_IMG_DIR = "/tmp/hm_images_cache" # Thư mục tạm trên Space

@st.cache_resource
def setup_image_cache():
    """Tải và giải nén ảnh từ Hugging Face Dataset (Chỉ chạy 1 lần)"""
    if not os.path.exists(LOCAL_IMG_DIR):
        os.makedirs(LOCAL_IMG_DIR, exist_ok=True)
        try:
            print(" Đang tải kho ảnh từ Dataset (Lần đầu sẽ lâu)...")
            zip_path = hf_hub_download(
                repo_id=DATASET_REPO_ID,
                filename=ZIP_FILENAME,
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN")
            )
            
            print(" Đang giải nén...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(LOCAL_IMG_DIR)
            print("Kho ảnh đã sẵn sàng!")
            return True
        except Exception as e:
            print(f"❌ Lỗi tải ảnh: {e}")
            return False
    return True

# Kích hoạt hệ thống ảnh
cache_status = setup_image_cache()

# ============================================
# 3. LOAD MODEL & DATA
# ============================================
@st.cache_resource
def load_models():
    MODEL_PATH = "." 
    print("⏳ Loading Models & Data...")
    
    # Load DataFrame
    with open(f'{MODEL_PATH}/df_products.pkl', 'rb') as f:
        df = pickle.load(f)
        
    # Load BM25 (Xử lý nếu thiếu)
    try:
        with open(f'{MODEL_PATH}/bm25_model.pkl', 'rb') as f:
            bm25 = pickle.load(f)
    except:
        bm25 = None
        
    # Load Embeddings
    embeddings = np.load(f'{MODEL_PATH}/sbert_embeddings.npy')
    
    # Load SBERT
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    return df, bm25, embeddings, sbert_model

try:
    df, bm25, embeddings, sbert_model = load_models()
    
    # Build FAISS Index
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
except Exception as e:
    st.error(f"❌ Lỗi load model: {e}")
    st.stop()

# ============================================
# 4. CLASS SEARCH ENGINE & RECOMMENDATION
# ============================================
class ShopSearchEngine:
    def __init__(self, df, bm25, index, sbert_model, embeddings):
        self.df = df
        self.bm25 = bm25
        self.index = index
        self.sbert_model = sbert_model
        self.embeddings = embeddings # Lưu embeddings để dùng cho recommend
        
        self.phrase_synonyms = {
            'running shoes': ['trainers', 'sneakers', 'runners', 'athletic footwear'],
            'summer dress': ['sundress', 'floral dress', 'beachwear'],
            'hoodie': ['sweatshirt', 'hooded top'],
            'denim': ['jeans', 'blue jeans', 'trousers']
        }

    def _min_max_normalize(self, scores):
        min_s, max_s = np.min(scores), np.max(scores)
        if max_s - min_s == 0: return np.zeros_like(scores)
        return (scores - min_s) / (max_s - min_s)
    
    def _expand_query(self, query):
        q_lower = str(query).lower()
        terms = []
        for k, v in self.phrase_synonyms.items():
            if k in q_lower: terms.extend(v)
        if terms: return q_lower + " " + " ".join(list(set(terms)))
        return q_lower

    def search(self, query, top_k=20, alpha=0.5):
        # 1. Expand
        expanded_q = self._expand_query(query)
        
        # 2. Semantic Search
        q_vec = self.sbert_model.encode([query]).astype('float32')
        faiss.normalize_L2(q_vec)
        D, I = self.index.search(q_vec, len(self.df))
        
        sbert_raw = np.zeros(len(self.df))
        sbert_raw[I[0]] = D[0]
        sbert_norm = self._min_max_normalize(sbert_raw)
        
        # 3. Lexical Search
        if self.bm25:
            q_tok = re.sub(r"[^a-z0-9\s]", " ", expanded_q).split()
            bm25_raw = self.bm25.get_scores(q_tok)
            bm25_norm = self._min_max_normalize(bm25_raw)
            final_scores = (alpha * bm25_norm) + ((1 - alpha) * sbert_norm)
        else:
            final_scores = sbert_norm
            bm25_norm = np.zeros(len(self.df))
            
        # 4. Sort & Format
        top_indices = np.argsort(final_scores)[::-1][:top_k]
        results = self.df.iloc[top_indices].copy()
        results['score'] = final_scores[top_indices]
        return results, expanded_q

    def get_related_products(self, article_id, top_k=5):
        """Gợi ý sản phẩm tương tự dựa trên vector"""
        try:
            # Tìm index của sản phẩm trong dataframe
            idx = self.df[self.df['article_id'].astype(str) == str(article_id)].index[0]
            
            # Lấy vector của nó
            target_vec = self.embeddings[idx].reshape(1, -1).astype('float32')
            faiss.normalize_L2(target_vec)
            
            # Search (Lấy top_k + 1 vì kết quả đầu tiên là chính nó)
            D, I = self.index.search(target_vec, top_k + 1)
            
            # Bỏ qua kết quả đầu tiên (chính nó)
            related_indices = I[0][1:]
            related_products = self.df.iloc[related_indices].copy()
            related_products['score'] = D[0][1:]
            
            return related_products
        except:
            return None

engine = ShopSearchEngine(df, bm25, index, sbert_model, embeddings)

# ============================================
# 5. QUẢN LÝ TRẠNG THÁI (SESSION STATE)
# ============================================
if 'selected_product_id' not in st.session_state:
    st.session_state.selected_product_id = None

def view_product(aid):
    st.session_state.selected_product_id = str(aid)

def back_to_search():
    st.session_state.selected_product_id = None

# Helper để lấy đường dẫn ảnh
def get_img_path(aid):
    aid_str = str(aid).zfill(10)
    path = os.path.join(LOCAL_IMG_DIR, f"{aid_str}.jpg")
    if os.path.exists(path):
        return path
    return "https://via.placeholder.com/300x400.png?text=No+Image"

# ============================================
# 6. GIAO DIỆN CHÍNH (UI)
# ============================================

# --- MÀN HÌNH CHI TIẾT SẢN PHẨM ---
if st.session_state.selected_product_id:
    aid = st.session_state.selected_product_id
    
    # Header & Nút Back
    c_back, c_title = st.columns([1, 5])
    with c_back:
        st.button("⬅️ Quay lại", on_click=back_to_search)
    
    try:
        # Lấy thông tin
        prod = df[df['article_id'].astype(str) == aid].iloc[0]
        
        # Layout 2 cột: Ảnh - Thông tin
        c_img, c_info = st.columns([1.5, 3])
        
        with c_img:
            st.image(get_img_path(aid), use_container_width=True)
            
        with c_info:
            st.title(prod['prod_name'])
            st.markdown(f"### ${prod.get('price', 0):.2f}")
            st.write(f"**Màu sắc:** {prod.get('colour_group_name', 'N/A')}")
            st.write(f"**Danh mục:** {prod.get('product_type_name', 'N/A')}")
            st.info(prod.get('detail_desc', 'Chưa có mô tả chi tiết.'))
            st.button("🛒 Thêm vào giỏ hàng", key="add_to_cart")
            st.caption(f"ID: {aid}")

        st.divider()
        st.subheader("🔍 Sản phẩm tương tự (Có thể bạn sẽ thích)")
        
        # Phần Recommendation
        related = engine.get_related_products(aid, top_k=5)
        if related is not None:
            cols = st.columns(5)
            for idx, (i, row) in enumerate(related.iterrows()):
                r_aid = str(row['article_id']).zfill(10)
                with cols[idx]:
                    st.image(get_img_path(r_aid), use_container_width=True)
                    st.caption(f"{row['prod_name'][:20]}...")
                    # Nút xem tiếp
                    st.button("Xem", key=f"rec_{r_aid}", on_click=view_product, args=(r_aid,))
                    
    except Exception as e:
        st.error("Không tìm thấy thông tin sản phẩm.")
        if st.button("Reset"): back_to_search()

# --- MÀN HÌNH TÌM KIẾM (TRANG CHỦ) ---
else:
    st.title("H&M AI Fashion Search")
    st.caption("Tìm kiếm thông minh với Hybrid Search & Recommendation")
    
    # Sidebar Config
    with st.sidebar:
        st.header(" Bộ lọc")
        alpha = st.slider("Alpha (Semantic vs Keyword)", 0.0, 1.0, 0.5)
        top_k = st.slider("Số kết quả hiển thị", 5, 20, 10)
        st.markdown("---")
        st.info(" Thử tìm: 'Black running shoes', 'Floral summer dress'...")

    # Search Box
    c_input, c_btn = st.columns([4, 1])
    with c_input:
        query = st.text_input("Bạn đang tìm gì?", placeholder="Mô tả sản phẩm...", key="search_box")
    with c_btn:
        st.write("")
        st.write("")
        do_search = st.button("🔍 Tìm kiếm")

    if do_search or query:
        with st.spinner("AI đang tìm kiếm..."):
            results, expanded_q = engine.search(query, top_k=top_k, alpha=alpha)
        
        # # Debug Info
        # with st.expander("🕵️‍♂️ Xem cơ chế AI (Debug)"):
        #     st.write(f"**Query gốc:** {query}")
        #     if query.lower() != expanded_q:
        #         st.success(f"**Expanded:** {expanded_q}")
        #     else:
        #         st.info("Query giữ nguyên.")

        st.markdown(f"### Tìm thấy {len(results)} kết quả phù hợp")
        
        # Vòng lặp hiển thị kết quả
        for idx, row in results.iterrows():
            with st.container():
                c1, c2, c3 = st.columns([1.5, 4.5, 1.5])
                
                # Lấy ID an toàn
                raw_id = row.get('article_id', idx)
                aid_str = str(raw_id).zfill(10)
                
                with c1:
                    st.image(get_img_path(aid_str), width=150)
                
                with c2:
                    st.subheader(row.get('prod_name', 'Unknown'))
                    st.write(f"**Giá:** ${row.get('price', 0):.2f}")
                    desc = str(row.get('detail_desc', ''))
                    st.write(desc[:200] + "..." if len(desc) > 200 else desc)
                    st.caption(f"ID: {aid_str}")
                
                with c3:
                    score = row.get('score', 0)
                    st.metric("Match Score", f"{score:.2f}")
                    # Nút Xem Chi Tiết -> Gọi hàm chuyển view
                    st.button("Xem chi tiết", key=f"main_{aid_str}", on_click=view_product, args=(aid_str,))
            
            st.divider()
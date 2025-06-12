import os
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import PyPDF2
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle
from typing import List, Dict, Tuple
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import hashlib
import fitz  # PyMuPDF برای نمایش PDF
import base64
from io import BytesIO

# تنظیمات لاگینگ
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# تنظیمات - کلید API خود را اینجا وارد کنید
GEMINI_API_KEY = "AIzaSyBAsYZit2n27gONsG2OTGwWNNSY9IuZtII"  # کلید API خود را اینجا قرار دهید
genai.configure(api_key=GEMINI_API_KEY)

PDF_FOLDER = "pdfs"  # پوشه‌ای که PDF ها در آن قرار دارند
VECTOR_DB_PATH = "vector_db"

# بارگذاری مدل embedding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

class PDFSearchEngine:
    def __init__(self):
        self.documents = []  # ذخیره متن اسناد
        self.document_metadata = []  # ذخیره metadata اسناد
        self.vector_index = None
        self.embeddings = None
        self.last_update = None
        
        # ایجاد پوشه‌های مورد نیاز
        os.makedirs(PDF_FOLDER, exist_ok=True)
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        
        # بارگذاری یا ایجاد ایندکس
        self.load_or_create_index()
        
        # راه‌اندازی file watcher
        self.setup_file_watcher()
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """استخراج متن از PDF به همراه شماره صفحه"""
        pages_data = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text and text.strip():
                        # تقسیم متن به پاراگراف‌ها
                        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
                        
                        for para in paragraphs:
                            if len(para) > 100:  # فقط پاراگراف‌های معنادار
                                pages_data.append({
                                    'text': para,
                                    'filename': os.path.basename(pdf_path),
                                    'page': page_num,
                                    'file_path': pdf_path
                                })
        except Exception as e:
            logger.error(f"خطا در استخراج متن از {pdf_path}: {e}")
        
        return pages_data
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """ایجاد embedding برای متن‌ها"""
        return embedding_model.encode(texts)
    
    def build_vector_index(self):
        """ساخت ایندکس برداری"""
        logger.info("در حال ساخت ایندکس برداری...")
        
        # پاک کردن داده‌های قبلی
        self.documents = []
        self.document_metadata = []
        
        # پردازش تمام PDF ها
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            pages_data = self.extract_text_from_pdf(pdf_path)
            
            for page_data in pages_data:
                self.documents.append(page_data['text'])
                self.document_metadata.append({
                    'filename': page_data['filename'],
                    'page': page_data['page'],
                    'file_path': page_data['file_path']
                })
        
        if not self.documents:
            logger.warning("هیچ سندی یافت نشد!")
            return
        
        # ایجاد embedding ها
        logger.info(f"در حال ایجاد embedding برای {len(self.documents)} سند...")
        self.embeddings = self.create_embeddings(self.documents)
        
        # ساخت ایندکس FAISS
        dimension = self.embeddings.shape[1]
        self.vector_index = faiss.IndexFlatIP(dimension)  # Inner Product برای شباهت
        
        # نرمال سازی برای استفاده از cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.vector_index.add(self.embeddings)
        
        self.last_update = datetime.now()
        logger.info(f"ایندکس برداری با موفقیت ساخته شد. تعداد اسناد: {len(self.documents)}")
        
        # ذخیره ایندکس
        self.save_index()
    
    def save_index(self):
        """ذخیره ایندکس در فایل"""
        try:
            # ذخیره ایندکس FAISS
            faiss.write_index(self.vector_index, os.path.join(VECTOR_DB_PATH, "index.faiss"))
            
            # ذخیره متادیتا
            with open(os.path.join(VECTOR_DB_PATH, "metadata.pkl"), "wb") as f:
                pickle.dump({
                    'documents': self.documents,
                    'document_metadata': self.document_metadata,
                    'last_update': self.last_update
                }, f)
            
            logger.info("ایندکس با موفقیت ذخیره شد")
        except Exception as e:
            logger.error(f"خطا در ذخیره ایندکس: {e}")
    
    def load_index(self):
        """بارگذاری ایندکس از فایل"""
        try:
            index_path = os.path.join(VECTOR_DB_PATH, "index.faiss")
            metadata_path = os.path.join(VECTOR_DB_PATH, "metadata.pkl")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                # بارگذاری ایندکس FAISS
                self.vector_index = faiss.read_index(index_path)
                
                # بارگذاری متادیتا
                with open(metadata_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.document_metadata = data['document_metadata']
                    self.last_update = data['last_update']
                
                # بازسازی embeddings
                self.embeddings = self.create_embeddings(self.documents)
                faiss.normalize_L2(self.embeddings)
                
                logger.info(f"ایندکس با موفقیت بارگذاری شد. تعداد اسناد: {len(self.documents)}")
                return True
        except Exception as e:
            logger.error(f"خطا در بارگذاری ایندکس: {e}")
        
        return False
    
    def load_or_create_index(self):
        """بارگذاری یا ایجاد ایندکس"""
        if not self.load_index():
            logger.info("ایندکس یافت نشد، در حال ساخت ایندکس جدید...")
            self.build_vector_index()
    
    def search_similar_documents(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict, float]]:
        """جستجوی اسناد مشابه"""
        if not self.vector_index or not self.documents:
            return []
        
        # ایجاد embedding برای query
        query_embedding = self.create_embeddings([query])
        faiss.normalize_L2(query_embedding)
        
        # جستجو در ایندکس
        scores, indices = self.vector_index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1:  # ایندکس معتبر
                results.append((
                    self.documents[idx],
                    self.document_metadata[idx],
                    float(score)
                ))
        
        return results

    def generate_answer_with_gemini(self, query: str, context_docs: List[Tuple[str, Dict, float]]) -> str:
        """تولید پاسخ با استفاده از Gemini"""
        if not context_docs:
            return "متأسفانه پاسخی در اسناد موجود یافت نشد."

        # ساخت context از اسناد مشابه
        context = "\n\n---\n\n".join([doc[0] for doc in context_docs[:3]])  # فقط 3 سند اول
        
        prompt = f"""شما یک دستیار پاسخگو به پرسش‌های مبتنی بر اسناد هستید.
فقط با استفاده از اطلاعات موجود در اسناد زیر به این سوال پاسخ دهید. 
اگر پاسخی در اسناد موجود نیست، بگویید «پاسخی در اسناد موجود یافت نشد».

سوال: {query}

اسناد:
{context}

پاسخ:"""
        
        try:
            # تست کردن مدل‌های مختلف Gemini
            models_to_try = [
                'gemini-1.5-flash',
                'gemini-1.5-pro', 
                'gemini-pro',
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro'
            ]
            
            for model_name in models_to_try:
                try:
                    logger.info(f"تلاش با مدل: {model_name}")
                    model = genai.GenerativeModel(model_name)
                    response = model.generate_content(prompt)
                    logger.info(f"موفقیت با مدل: {model_name}")
                    return response.text.strip()
                except Exception as model_error:
                    logger.warning(f"مدل {model_name} کار نکرد: {model_error}")
                    continue
            
            # اگر هیچ مدلی کار نکرد، پاسخ پیش‌فرض
            return "خطا در دسترسی به مدل Gemini. لطفاً کلید API و اتصال اینترنت را بررسی کنید."
            
        except Exception as e:
            logger.error(f"خطا در تولید پاسخ با Gemini: {e}")
            return f"خطایی در تولید پاسخ رخ داد: {str(e)}"
    
    def search(self, query: str) -> Dict:
        """جستجوی اصلی"""
        # یافتن اسناد مشابه
        similar_docs = self.search_similar_documents(query, top_k=5)
        
        if not similar_docs:
            return {
                "answer": "متأسفانه پاسخی در اسناد موجود یافت نشد.",
                "sources": []
            }
        
        # تولید پاسخ
        answer = self.generate_answer_with_gemini(query, similar_docs)
        
        # استخراج منابع
        sources = []
        seen_sources = set()
        
        for doc, metadata, score in similar_docs:
            source_key = f"{metadata['filename']}_{metadata['page']}"
            if source_key not in seen_sources and score > 0.3:  # فقط منابع با امتیاز بالا
                sources.append({
                    "filename": metadata['filename'],
                    "page": metadata['page'],
                    "score": score
                })
                seen_sources.add(source_key)
        
        return {
            "answer": answer,
            "sources": sources[:5]  # حداکثر 5 منبع
        }
    
    def setup_file_watcher(self):
        """راه‌اندازی نظارت بر تغییرات فایل‌ها"""
        class PDFHandler(FileSystemEventHandler):
            def __init__(self, search_engine):
                self.search_engine = search_engine
                self.last_modification = time.time()
            
            def on_modified(self, event):
                if not event.is_directory and event.src_path.lower().endswith('.pdf'):
                    # جلوگیری از پردازش مکرر
                    if time.time() - self.last_modification > 5:
                        logger.info(f"PDF جدید یا تغییر یافته شناسایی شد: {event.src_path}")
                        threading.Thread(target=self.search_engine.build_vector_index).start()
                        self.last_modification = time.time()
            
            def on_created(self, event):
                self.on_modified(event)
        
        event_handler = PDFHandler(self)
        observer = Observer()
        observer.schedule(event_handler, PDF_FOLDER, recursive=False)
        observer.start()
        
        logger.info("File watcher راه‌اندازی شد")

# ایجاد نمونه موتور جستجو
search_engine = PDFSearchEngine()

@app.route('/')
def index():
    """صفحه اصلی"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "فایل index.html یافت نشد", 404

@app.route('/api/search', methods=['POST'])
def api_search():
    """API جستجو"""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({"error": "سوال خالی است"}), 400
        
        logger.info(f"جستجو برای: {query}")
        
        # انجام جستجو
        result = search_engine.search(query)
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"خطا در API جستجو: {e}")
        return jsonify({"error": f"خطای داخلی سرور: {str(e)}"}), 500

@app.route('/pdf-viewer')
def pdf_viewer():
    """نمایشگر PDF - فقط مشاهده بدون دانلود"""
    filename = request.args.get('file')
    page = request.args.get('page', 1, type=int)
    
    if not filename:
        return "فایل مشخص نشده", 400
    
    pdf_path = os.path.join(PDF_FOLDER, filename)
    
    if not os.path.exists(pdf_path):
        return "فایل یافت نشد", 404
    
    try:
        # تبدیل صفحه PDF به تصویر
        doc = fitz.open(pdf_path)
        if page > len(doc) or page < 1:
            return "شماره صفحه نامعتبر", 400
            
        page_obj = doc[page - 1]
        pix = page_obj.get_pixmap(matrix=fitz.Matrix(2, 2))  # بزرگ‌نمایی 2x
        img_data = pix.tobytes("png")
        img_base64 = base64.b64encode(img_data).decode()
        
        doc.close()
        
        # HTML برای نمایش تصویر PDF
        viewer_html = f"""
        <!DOCTYPE html>
        <html lang="fa" dir="rtl">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>نمایش PDF - {filename}</title>
            <style>
                body {{ 
                    margin: 0; 
                    padding: 20px; 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: #f5f5f5;
                }}
                .header {{ 
                    background: white; 
                    padding: 20px; 
                    margin: -20px -20px 20px -20px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    border-radius: 0 0 10px 10px;
                }}
                .pdf-container {{ 
                    text-align: center; 
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .pdf-image {{ 
                    max-width: 100%; 
                    height: auto; 
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                }}
                .close-btn {{
                    background: #dc3545;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                }}
                .close-btn:hover {{
                    background: #c82333;
                }}
                .page-info {{
                    color: #666;
                    margin-bottom: 10px;
                }}
                .no-download {{
                    color: #28a745;
                    font-size: 14px;
                    margin-top: 10px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>📄 {filename}</h2>
                <div class="page-info">صفحه {page}</div>
                <button class="close-btn" onclick="window.close()">بستن</button>
                <div class="no-download">✅ فقط مشاهده - قابل دانلود نیست</div>
            </div>
            <div class="pdf-container">
                <img src="data:image/png;base64,{img_base64}" class="pdf-image" alt="PDF Page {page}">
            </div>
        </body>
        </html>
        """
        
        return viewer_html
        
    except Exception as e:
        logger.error(f"خطا در نمایش PDF: {e}")
        return f"خطا در نمایش PDF: {str(e)}", 500

@app.route('/api/status')
def api_status():
    """وضعیت سیستم"""
    return jsonify({
        "status": "active",
        "documents_count": len(search_engine.documents),
        "last_update": search_engine.last_update.isoformat() if search_engine.last_update else None,
        "pdf_folder": PDF_FOLDER,
        "gemini_configured": GEMINI_API_KEY != "YOUR_GEMINI_API_KEY_HERE"
    })

@app.route('/api/rebuild-index', methods=['POST'])
def api_rebuild_index():
    """بازسازی ایندکس"""
    try:
        threading.Thread(target=search_engine.build_vector_index).start()
        return jsonify({"message": "بازسازی ایندکس آغاز شد"})
    except Exception as e:
        logger.error(f"خطا در بازسازی ایندکس: {e}")
        return jsonify({"error": "خطا در بازسازی ایندکس"}), 500

# API جدید برای تست مدل‌های دردسترس
@app.route('/api/test-gemini', methods=['GET'])
def test_gemini():
    """تست مدل‌های Gemini موجود"""
    try:
        # لیست کردن مدل‌های موجود
        models = genai.list_models()
        available_models = []
        
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
        
        return jsonify({
            "status": "success",
            "available_models": available_models
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        })

if __name__ == '__main__':
    print("""
    🚀 سیستم جستجوی هوشمند PDF راه‌اندازی شد!
    
    📋 مراحل راه‌اندازی:
    1. کلید API Gemini را در GEMINI_API_KEY وارد کنید  
    2. فایل‌های PDF را در پوشه 'pdfs' قرار دهید
    3. requirements را نصب کنید:
       pip install flask flask-cors PyPDF2 google-generativeai sentence-transformers faiss-cpu watchdog PyMuPDF
    
    🌐 سرور در http://localhost:5000 در حال اجرا است
    🔒 PDF ها فقط قابل مشاهده هستند - دانلود غیرفعال است
    
    🔧 برای تست مدل‌های موجود: http://localhost:5000/api/test-gemini
    """)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
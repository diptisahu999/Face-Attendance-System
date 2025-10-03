# Face Recognition API

A FastAPI-based face recognition service that allows you to **upload employee images**, **store embeddings**, and **recognize faces in real time**.

---

## 🚀 Features
- Upload multiple employee images to generate embeddings  
- Recognize faces in uploaded photos  
- Store employee data in a database with SQLAlchemy  
- Maintain an in-memory embedding cache for fast recognition  
- Create recognition logs automatically  
- REST API with Swagger & ReDoc documentation  

---

## 🛠️ Installation

model : buffalo_l(glintr100.onnx)

### 1. Clone the repository
```bash
git clone https://github.com/your-username/face-recognition-api.git
cd face-recognition-api

python -m venv venv
source venv/bin/activate   # On Linux / Mac
venv\Scripts\activate      # On Windows

pip install -r requirements.txt


uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

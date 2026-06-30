import os
import fitz  # PyMuPDF (PDF'leri okumak için)
import pickle
from sentence_transformers import SentenceTransformer

def process_pdfs(folder_path):
    chunks = []
    print(f"'{folder_path}' klasöründeki PDF'ler okunuyor...")
    
    # Klasördeki tüm PDF'leri bul
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            print(f"İşleniyor: {file}")
            doc = fitz.open(os.path.join(folder_path, file))
            text = ""
            for page in doc: 
                text += page.get_text()
            
            # PDF'i 800 karakterlik küçük bilgi parçalarına böl (100 karakter üst üste binsin ki cümle bölünmesin)
            for i in range(0, len(text), 700):
                chunk = text[i:i + 800].replace('\n', ' ').strip()
                if len(chunk) > 100: 
                    chunks.append(chunk)
                    
    return chunks

# 1. PDF'leri parçala
chunks = process_pdfs("rehberler")
print(f"Toplam {len(chunks)} adet bilgi paragrafı çıkarıldı.")

# 2. Yapay Zeka ile bu metinleri "Vektör" (sayısal anlama) çevir
print("Tıbbi Dil Modeli yükleniyor (BAAI/bge-m3)... Bu işlem biraz sürebilir.")
search_model = SentenceTransformer('BAAI/bge-m3')

print("Paragraflar veritabanı için vektörleniyor...")
embeddings = search_model.encode(chunks, show_progress_bar=True)

# 3. Sonucu projenin içine (models klasörüne) kaydet
# Eğer 'models' klasörünüz yoksa önce onu oluşturun!
os.makedirs('models', exist_ok=True)

with open('models/rag_data.pkl', 'wb') as f:
    pickle.dump({'chunks': chunks, 'embeddings': embeddings}, f)
    
print("BAŞARILI! RAG Veritabanı başarıyla oluşturuldu: 'models/rag_data.pkl'")
print("Artık web siteniz bu veritabanından saniyeler içinde bilgi çekebilir.")
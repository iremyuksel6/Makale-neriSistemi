import fasttext
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import psutil

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory Usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

print_memory_usage()

model_path = r"C:\Users\Vahit Yüksel\Desktop\python\yazlab3\cc.en.300.bin"
model = fasttext.load_model(model_path)
makaleler_dizini = r"C:\Users\Vahit Yüksel\Desktop\python\yazlab3\dogal_dil_sonuc"
ilgilenilen_kelimeler = ["problem", "processing", "system", "graph", "routing"]
ortalama_vektorler = []

for dosya in os.listdir(makaleler_dizini):
    if dosya.endswith(".txt"):
        dosya_yolu = os.path.join(makaleler_dizini, dosya)
        with open(dosya_yolu, "r", encoding="utf-8") as dosya:
            makale_metni = dosya.read().replace('\n', ' ')
            vektor1 = model.get_sentence_vector(makale_metni)
            if any(kelime in makale_metni for kelime in ilgilenilen_kelimeler):
                vektor = model.get_sentence_vector(makale_metni)
                ortalama_vektorler.append(vektor)

ortalama_vektorler_np = np.array(ortalama_vektorler)
ortalama_vektor = np.mean(ortalama_vektorler_np, axis=0)
print("Ortalama Vektör:", ortalama_vektor)
tum_vektorler = np.array([model.get_sentence_vector(open(os.path.join(makaleler_dizini, dosya), "r", encoding="utf-8").read().replace('\n', ' ')) for dosya in os.listdir(makaleler_dizini) if dosya.endswith(".txt")])
benzerlikler = cosine_similarity(tum_vektorler, [ortalama_vektor])
benzerlik_listesi = [(dosya, benzerlik) for dosya, benzerlik in zip(os.listdir(makaleler_dizini), benzerlikler)]
benzerlik_listesi_sirali = sorted(benzerlik_listesi, key=lambda x: x[1], reverse=True)
for dosya, benzerlik in benzerlik_listesi_sirali[:5]:
    print(f"Dosya: {dosya}, Benzerlik: {benzerlik}")

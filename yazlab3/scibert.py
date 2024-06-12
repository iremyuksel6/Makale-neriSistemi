import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

model_name = "allenai/scibert_scivocab_uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

max_length = 15
makale_dizini = r"C:\Users\Vahit Yüksel\Desktop\python\yazlab3\dogal_dil_sonuc"
makale_vektorleri = {}

for dosya_adı in os.listdir(makale_dizini):
    if dosya_adı.endswith(".txt"):
        dosya_yolu = os.path.join(makale_dizini, dosya_adı)
        with open(dosya_yolu, "r", encoding="utf-8") as dosya:
            metin = dosya.read()
        tokenler = tokenizer(metin, return_tensors="pt", max_length=max_length, truncation=True, padding='max_length')
        with torch.no_grad():
            cikti = model(**tokenler)
        vektor = cikti.pooler_output[0].numpy()
        makale_vektorleri[dosya_adı] = vektor

vektor_dosyasi = r"C:\Users\Vahit Yüksel\Desktop\python\yazlab3\makale_vektorleri.npy"
with open(vektor_dosyasi, "wb") as dosya:
    np.save(dosya, makale_vektorleri)

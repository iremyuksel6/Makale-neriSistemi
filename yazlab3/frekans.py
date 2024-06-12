import os
from collections import Counter

dosya_yolu = r'C:\Users\Vahit Yüksel\Desktop\python\yazlab3\Krapivin2009\Krapivin2009\keys'
dosya_isimleri = os.listdir(dosya_yolu)
tum_kelimeler = []

for dosya_adi in dosya_isimleri:
    with open(os.path.join(dosya_yolu, dosya_adi), 'r', encoding='utf-8') as dosya:
        icerik = dosya.read()
        kelimeler = icerik.lower().split()
        tum_kelimeler.extend(kelimeler)

kelime_sayilari = Counter(tum_kelimeler)

for kelime, sayi in sorted(kelime_sayilari.items(), key=lambda x: x[1]):
    print(f"{kelime}: {sayi}")

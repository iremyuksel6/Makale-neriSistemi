import os
import nltk
import glob
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import fasttext

fModel = fasttext.load_model("cc.en.300.bin")

from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
model = AutoModelForMaskedLM.from_pretrained("allenai/scibert_scivocab_uncased")

nltk.download('punkt')
nltk.download('stopwords')

data_folder = r"C:\Users\Vahit Yüksel\Desktop\python\yazlab3\Krapivin2009\Krapivin2009"
docs_folder = os.path.join(data_folder, "docsutf8")
output_folder = r"C:\Users\Vahit Yüksel\Desktop\python\yazlab3\dogal_dil_sonuc"

stop_words = set(stopwords.words('english'))

porter = PorterStemmer()

punctuation = string.punctuation

for file_path in glob.glob(os.path.join(docs_folder, "*.txt")):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    text = text.lower()

    text = ''.join([char for char in text if char not in punctuation])

    words = word_tokenize(text)

    stemmed_words = [porter.stem(word) for word in words if word not in stop_words]

    output_file_path = os.path.join(output_folder, os.path.basename(file_path))
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        output_file.write(" ".join(stemmed_words))
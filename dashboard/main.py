import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from deep_translator import GoogleTranslator
import nltk
import nltk.data
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
import os
from dotenv import load_dotenv
import re
import chardet
from deep_translator.exceptions import RequestError
import sys
import os

# Tambahkan path direktori tempat 'utils' berada
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import get_model

# Download NLTK data
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
    print("VADER Lexicon ditemukan secara lokal. Tidak perlu mengunduh lagi.")
except LookupError:
    print("VADER Lexicon tidak ditemukan. Mengunduh...")
    nltk.download('vader_lexicon')

# Sekarang import SentimentIntensityAnalyzer setelah lexicon tersedia
from nltk.sentiment.vader import SentimentIntensityAnalyzer


load_dotenv()
hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")

# Base URL for API
BASE_URL = "http://localhost:8080"
BASE_ML_URL = "http://127.0.0.1:5000"


def splash_screen():
    
    st.markdown(
        """
        <style>
        .center-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 10vh; /* Membuat elemen berada di tengah vertikal */
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    st.title("Welcome to the Sentiment Result App")
    st.write("Analyze and preprocess Tokopedia product reviews effortlessly.")

    if st.button("Get Started"):
        st.session_state['page'] = "login"

def login_page():
    st.title("Login")
    st.write("Check if your product is good or not")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login", key="login_button"):
        response = api_login(email, password)
        if response.get("status"):
            st.session_state['authenticated'] = True
            st.session_state['token'] = response["data"]["token"]
            st.session_state['page'] = "Home"
        else:
            st.error(response.get("message", "Login failed"))
    
    if st.button("Don't have an account? Register here"):
        st.session_state['page'] = "register"
        st.rerun()


def register_page():
    st.title("Register")
    st.write("Start your product check with us")
    email = st.text_input("Email")
    name = st.text_input("Name")
    password = st.text_input("Password", type="password")
    if st.button("Register", key="register_button"):
        response = api_register(email, name, password)
        if response.get("status"):
            st.success("Registration successful! Please login.")
            st.session_state['page'] = "login"
        else:
            st.error(response.get("message", "Registration failed"))
    
    if st.button("Already have an account? Login here"):
        st.session_state['page'] = "login"
        st.rerun()
                
# Helper functions for API calls
def api_register(email, name, password):
    url = f"{BASE_URL}/api/user"
    payload = {
        "email": email,
        "name": name,
        "password": password
    }
    response = requests.post(url, json=payload)
    return response.json()

def api_login(email, password):
    url = f"{BASE_URL}/api/user/login"
    payload = {
        "email": email,
        "password": password
    }
    response = requests.post(url, json=payload)
    return response.json()

def api_predict(statements):
    url = f"{BASE_ML_URL}/predict"
    payload = {"statements": statements}
    headers = {"api-key": hugging_face_token}
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

def api_product_result(product_url):
    url = f"{BASE_URL}/api/ml/guest/analysis"
    params = {"product_url": product_url}
    response = requests.get(url, params=params)
    return response.json()


def dataset_page():
    st.title("Dataset")
    uploaded_file = st.file_uploader("Upload your CSV file for preprocessing and result:", type=["csv"], key="csv_uploader")

    if uploaded_file is not None:
        # Deteksi encoding file
        raw_data = uploaded_file.read()
        detected_encoding = chardet.detect(raw_data)['encoding']
        
        # Jika encoding tidak ditemukan, fallback ke utf-8 atau latin-1
        if detected_encoding is None:
            detected_encoding = "utf-8"

        st.write(f"**Detected Encoding:** {detected_encoding}")

        # Reset posisi pembacaan file ke awal setelah membaca untuk deteksi encoding
        uploaded_file.seek(0)

        try:
            # Coba membaca CSV dengan encoding yang terdeteksi
            df = pd.read_csv(uploaded_file, encoding=detected_encoding, low_memory=False)
        except UnicodeDecodeError:
            # Jika tetap error, gunakan encoding "latin-1" sebagai fallback
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding="latin-1", low_memory=False)
        except pd.errors.EmptyDataError:
            # Jika file kosong, tampilkan pesan error
            st.error("The uploaded file is empty or contains only headers. Please upload a valid dataset.")
            return

        # Periksa apakah ada kolom di dalam CSV
        if df.empty:
            st.error("The uploaded CSV file is empty. Please upload a file with data.")
            return

        # Tampilkan preview dataset
        st.write("**Uploaded Dataset Preview:**")
        st.write(df.head())

        # Validasi apakah kolom 'review' dan 'rating' ada dalam dataset
        if "review" not in df.columns or "rating" not in df.columns:
            st.error("The CSV must contain 'review' and 'rating' columns.")
            return

        # Membersihkan dataset: hapus duplikat dan baris kosong
        df = df.drop_duplicates()
        df = df.dropna(subset=["review", "rating"])

        # Menampilkan hasil setelah pembersihan
        st.write("**Cleaned Dataset Preview:**")
        st.write(df.head())

        # Simpan dataset ke dalam session state
        st.session_state["processed_data"] = df
        st.success("Dataset loaded and cleaned successfully! You can now proceed to the Preprocessing page.")

        # if st.button("Go to Preprocessing"):
        #     st.session_state['page'] = "Preprocessing"

def load_normalization_map(file_path):
    df = pd.read_csv(file_path, sep=";", header=None, names=["abbreviation", "normal"])
    return {row["abbreviation"]: row["normal"] for _, row in df.iterrows()}

def preprocessing_page():
    st.title("Preprocessing")
    st.write("Perform data preprocessing and view the step-by-step process.")

    # Validasi apakah data dari dataset telah dimuat
    if "processed_data" not in st.session_state:
        st.warning("No dataset found. Please upload data on the Dataset page first.")
        return

    df = st.session_state["processed_data"]
    
    st.write("**Dataset Preview:**")
    st.write(df.head())

    if st.button("Start Preprocessing", key="start_preprocessing"):
        normalization_map = load_normalization_map("./kamus/kamus_singkatan.csv")

        # Proses Preprocessing
        st.write("**Original Dataset Preview (Top 5):**")
        st.write(df.head())

        df = preprocess_data(df, normalization_map)
        st.write("**1. Preprocessing Steps:**")
        st.write(df[["case_folded", "cleaned", "normalized", "filtered", "stemmed", "tokenized", "preprocessed_review"]].head())
        st.write(df.head())

        df = label_data_with_vader(df)
        st.write("**2. Labeling Data with VADER:**")
        st.write(df.head())
        # st.write(df)

        st.session_state["preprocessed_data"] = df
        st.success("Preprocessing complete! You can now proceed to the Result page.")
        # if st.button("Go to Result"):
        #     st.session_state['page'] = "Result"


def preprocess_data(df, normalization_map):
    # Initialize Sastrawi stopword remover and stemmer
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    stemmer = StemmerFactory().create_stemmer()
    
    def clean_text(text):
        """ Membersihkan teks dari emoji, hashtag, username, URL, dan spasi berlebih """
        text = re.sub(r'http\S+|www\S+', '', text)  # Hapus URL
        text = re.sub(r'@\w+', '', text)  # Hapus username (@username)
        text = re.sub(r'#\w+', '', text)  # Hapus hashtag (#hashtag)
        text = re.sub(r'[^\w\s,]', '', text)  # Hapus karakter selain huruf, angka, spasi, dan koma
        text = re.sub(r'\s+', ' ', text).strip()  # Hapus spasi ganda dan spasi di awal/akhir
        return text

    # Langkah preprocessing per teks
    def preprocess_text_steps(text):
        steps = {}
        # Step 1: Case Folding
        case_folded_text = text.lower()
        steps["case_folded"] = case_folded_text
        # Step 2: Cleaning
        cleaned_text = clean_text(case_folded_text)
        steps["cleaned"] = cleaned_text
        # Step 3: Normalization
        normalized_text = " ".join([normalization_map.get(word, word) for word in cleaned_text.split()])
        steps["normalized"] = normalized_text
        # Step 4: Filtering (Stopword Removal)
        filtered_text = stopword_remover.remove(normalized_text)
        steps["filtered"] = filtered_text
        # Step 5: Stemming
        stemmed_text = " ".join([stemmer.stem(word) for word in filtered_text.split()])  # Stem setiap kata
        steps["stemmed"] = stemmed_text
        # Step 6: Tokenizing (Dilakukan setelah stemming)
        tokenized_text = stemmed_text.split()  # Tokenisasi ke dalam bentuk list
        steps["tokenized"] = tokenized_text

        return steps

    # Terapkan preprocessing dan simpan langkah-langkah
    preprocessing_steps = df["review"].apply(preprocess_text_steps)

    # Tambahkan setiap langkah ke dalam DataFrame
    df["case_folded"] = preprocessing_steps.apply(lambda x: x["case_folded"])
    df["cleaned"] = preprocessing_steps.apply(lambda x: x["cleaned"])
    df["normalized"] = preprocessing_steps.apply(lambda x: x["normalized"])
    df["filtered"] = preprocessing_steps.apply(lambda x: x["filtered"])
    df["stemmed"] = preprocessing_steps.apply(lambda x: x["stemmed"])
    df["tokenized"] = preprocessing_steps.apply(lambda x: x["tokenized"])

    # Tambahkan kolom final "preprocessed_review"
    df["preprocessed_review"] = df["stemmed"]  # Preprocessed review tetap dalam format string

    return df




def label_data_with_vader(df):
    sid = SentimentIntensityAnalyzer()
    translation_cache = {}  # Cache untuk teks yang sudah diterjemahkan

    def label_sentiment(text):
        # Gunakan cache jika teks sudah pernah diterjemahkan
        if text in translation_cache:
            english_text = translation_cache[text]
        else:
            try:
                english_text = GoogleTranslator(source='auto', target='en').translate(text)
                translation_cache[text] = english_text  # Simpan hasil terjemahan ke cache
            except RequestError:
                st.warning("Google Translator API error! Using original text.")
                english_text = text  # Jika gagal, gunakan teks asli

        score = sid.polarity_scores(english_text)
        if score['compound'] >= 0.05:
            return 1  # Positive
        elif score['compound'] <= -0.05:
            return 0  # Negative
        else:
            return -1  # Neutral

    df['label'] = df['review'].apply(label_sentiment)
    df['label'] = df['label'].replace(-1, 0)

    # Validasi distribusi label
    st.write("**Label Distribution:**")
    st.write(df['label'].value_counts())

    return df


def train_and_evaluate_model(df):
    model_name = 'cahya/bert-base-indonesian-522M'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    MAX_LENGTH = 100

    # Validasi distribusi label
    label_counts = df['label'].value_counts()
    st.write("**Label Distribution Before Splitting:**")
    st.write(label_counts)

    # Tangani kelas dengan kurang dari dua contoh
    if label_counts.min() < 2:
        st.error("One of the classes has less than 2 samples, which is too few for stratified splitting.")
        return None, None

    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(
        df['preprocessed_review'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    )
    # X_train, X_test, y_train, y_test = train_test_split(
    #     df['review'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
    # )

    # Tokenization
    train_tokenized = tokenizer(
        text=X_train.tolist(),
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors='tf'
    )

    test_tokenized = tokenizer(
        text=X_test.tolist(),
        add_special_tokens=True,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors='tf'
    )

    train_input_ids = train_tokenized['input_ids']
    train_attention_mask = train_tokenized['attention_mask']
    test_input_ids = test_tokenized['input_ids']
    test_attention_mask = test_tokenized['attention_mask']

    # Model
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Training
    history = model.fit(
        [train_input_ids, train_attention_mask],
        y_train,
        validation_data=([test_input_ids, test_attention_mask], y_test),
        epochs=3,
        batch_size=16
    )

    # Evaluation
    predictions = model.predict([test_input_ids, test_attention_mask])
    predicted_labels = tf.argmax(predictions.logits, axis=1)

    metrics = {
        "Accuracy": accuracy_score(y_test, predicted_labels),
        "Precision": precision_score(y_test, predicted_labels, average='weighted'),
        "Recall": recall_score(y_test, predicted_labels, average='weighted'),
        "F1-Score": f1_score(y_test, predicted_labels, average='weighted')
    }

    # Confusion Matrix
    cm = confusion_matrix(y_test, predicted_labels)
    return model, metrics, cm

            


def predict_with_trained_model(model, tokenizer, text):
    """
    Prediksi sentimen berdasarkan model yang sudah dilatih.
    Args:
        model: Model TFBertForSequenceClassification yang sudah dilatih.
        tokenizer: Tokenizer BERT yang sesuai dengan model.
        text: Teks input untuk prediksi sentimen.
    Returns:
        Sentimen prediksi ('Positive' atau 'Negative').
    """
    tokenized_input = tokenizer(
        text=[text],
        add_special_tokens=True,
        max_length=100,
        truncation=True,
        padding="max_length",
        return_tensors="tf"
    )

    prediction = model.predict([tokenized_input["input_ids"], tokenized_input["attention_mask"]])
    predicted_label = tf.argmax(prediction.logits, axis=1).numpy()[0]

    if predicted_label == 1:
        return "Positive"
    else:
        return "Negative"


def result_page():
    st.title("Result")

    if "preprocessed_data" not in st.session_state:
        st.warning("No preprocessed data found. Please complete preprocessing first.")
        return

    df = st.session_state["preprocessed_data"].copy()
    
    st.write("**Preprocessed Dataset Preview:**")
    st.write(df.head())


    if st.button("Train and Evaluate Model"):
        model, metrics, cm = train_and_evaluate_model(df)

        st.write("### Evaluation Metrics")
        for key, value in metrics.items():
            st.write(f"**{key}:** {value:.2f}")

        # Display Confusion Matrix
        st.write("### Confusion Matrix")
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(ax=ax)
        st.pyplot(fig)

        st.session_state['model'] = model
        st.session_state['metrics'] = metrics
        st.session_state['confusion_matrix'] = cm

    if "metrics" in st.session_state:
        st.write("### Last Trained Model Metrics")
        for key, value in st.session_state['metrics'].items():
            st.write(f"**{key}:** {value:.2f}")
        

    if "model" in st.session_state:
        st.write("### Test Model")
        test_input = st.text_area("Enter text for sentiment prediction:")

        if st.button("Predict"):
            
            # JANGAN DI UBAH
            # model = st.session_state['model']
            model = get_model('transformers-bert')
            tokenizer = BertTokenizer.from_pretrained('cahya/bert-base-indonesian-522M')

            tokenized_input = tokenizer(
                text=[test_input],
                add_special_tokens=True,
                max_length=100,
                truncation=True,
                padding='max_length',
                return_tensors='tf'
            )
            
            prediction = model.predict([tokenized_input['input_ids'], tokenized_input['attention_mask']])
            probabilities = tf.nn.softmax(prediction.logits, axis=1).numpy()[0]
            predicted_label = tf.argmax(prediction.logits, axis=1).numpy()[0]
            sentiment = "Positive" if predicted_label == 1 else "Negative"
            st.write(f"**Probability:** Positive: {probabilities[1]:.2f}, Negative: {probabilities[0]:.2f}")
            st.write(f"**Predicted Sentiment:** {sentiment}")

            # prediction = model.predict([tokenized_input['input_ids'], tokenized_input['attention_mask']])
            # sentiment = "Positive" if tf.argmax(prediction.logits, axis=1).numpy()[0] == 1 else "Negative"
            # st.write(f"**Predicted Sentiment:** {sentiment}")

        
def dashboard():
    # Navigasi antar halaman menggunakan session_state
    current_page = st.session_state.get("page", "Home")

    with st.sidebar:
        selected = option_menu(
            menu_title="Sentiment Analysis Of Tokopedia Product Reviews",
            options=["Home", "Dataset", "Preprocessing", "Result", "Search Product & Proses", "Logout"],
            icons=["house", "table", "gear", "graph-up", "search", "box-arrow-right"],
            menu_icon="cast",
            # default_index=["Home", "Dataset", "Preprocessing", "Result", "Search Product & Proses", "Logout"].index(current_page)
            default_index=0
        )

    # Perbarui halaman saat ini di session_state
    # st.session_state["page"] = selected

    if selected == "Home":
        st.title("Home")
        st.write("This application is used for sentiment result of Tokopedia product reviews.")
        st.write("**Main Features:**")
        st.write("1. Preprocess raw data into result-ready data.")
        st.write("2. Train and evaluate sentiment result models.")
        st.write("3. Analyze Tokopedia product reviews using advanced tools.")

    elif selected == "Dataset":
        dataset_page()

    elif selected == "Preprocessing":
        preprocessing_page()

    elif selected == "Result":
        result_page()

    elif selected == "Search Product & Proses":
        st.title("Search Product & Proses")
        product_url = st.text_input("Enter Tokopedia product link:", key="product_url_input")

        if st.button("Search Product & Proses", key="search_product_button"):
            if product_url:
                response = api_product_result(product_url)
                if response.get("status"):
                    data = response["data"]

                    st.header("Product Overview")
                    st.image(data["image_urls"][0], width=300)
                    st.write(f"**Product Name:** {data['product_name']}")
                    st.write(f"**Description:** {data['product_description']}")
                    st.write(f"**Rating:** {data['rating']} ({data['bintang']} stars)")
                    st.write(f"**Reviews:** {data['ulasan']}")
                    st.write(f"**Shop:** {data['shop_name']}")

                    st.header("Product Sentiment Result")
                    st.write(f"**Sentiment Summary:** {data['summary']}")
                    st.write(f"**Positive Reviews:** {data['count_positive']}")
                    st.write(f"**Negative Reviews:** {data['count_negative']}")

                    labels = ["Positive", "Negative"]
                    sizes = [data["count_positive"], data["count_negative"]]
                    fig, ax = plt.subplots()
                    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
                    ax.axis("equal")
                    st.pyplot(fig)
                else:
                    st.error("Failed to fetch product data. Please check the link.")

    elif selected == "Logout":
        st.session_state["authenticated"] = False
        st.session_state["page"] = "login"
        st.write("Logging out...")
        st.stop()



if __name__ == "__main__":
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if "page" not in st.session_state:
        st.session_state["page"] = "splash"

    if st.session_state["page"] == "splash":
        splash_screen()
    elif st.session_state["page"] == "login":
        login_page()
    elif st.session_state["page"] == "register":
        register_page()
    else:
        dashboard()
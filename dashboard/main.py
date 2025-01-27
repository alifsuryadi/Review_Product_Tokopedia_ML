import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os
from dotenv import load_dotenv


load_dotenv()
hugging_face_token = os.getenv("HUGGING_FACE_TOKEN")
# Base URL for API
BASE_URL = "http://localhost:8080"
BASE_ML_URL = "http://127.0.0.1:5000"

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

def api_product_analysis(product_url):
    url = f"{BASE_URL}/api/ml/guest/analysis"
    params = {"product_url": product_url}
    response = requests.get(url, params=params)
    return response.json()

def login_page():
    st.title("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    if st.button("Login", key="login_button"):
        response = api_login(email, password)
        if response.get("status"):
            st.session_state['authenticated'] = True
            st.session_state['token'] = response["data"]["token"]
            st.session_state['page'] = "dashboard"
        else:
            st.error(response.get("message", "Login failed"))

def register_page():
    st.title("Register")
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

def load_normalization_map(file_path):
    df = pd.read_csv(file_path, sep=";", header=None, names=["abbreviation", "normal"])
    return {row["abbreviation"]: row["normal"] for _, row in df.iterrows()}

def preprocessing_page():
    st.title("Preprocessing")
    st.write("Perform data preprocessing and view the step-by-step process.")

    # Load normalization map from file
    normalization_map = load_normalization_map("./kamus/kamus_singkatan.csv")

    # Initialize Sastrawi stopword remover and stemmer
    stopword_remover = StopWordRemoverFactory().create_stop_word_remover()
    stemmer = StemmerFactory().create_stemmer()

    raw_text = st.text_area("Enter raw data (separate sentences with commas):", key="preprocessing_text_area")

    if st.button("Start Preprocessing", key="start_preprocessing"):
        if raw_text:
            raw_data = [s.strip() for s in raw_text.split(",")]

            # Step 1: Cleaning
            cleaned_data = [s.replace("...", "").replace("!", "").replace(".", "") for s in raw_data]
            st.write("**1. Cleaning:**")
            st.write(cleaned_data)

            # Step 2: Case Folding
            case_folded_data = [s.lower() for s in cleaned_data]
            st.write("**2. Case Folding:**")
            st.write(case_folded_data)

            # Step 3: Normalization
            normalized_data = [" ".join([normalization_map.get(word, word) for word in s.split()]) for s in case_folded_data]
            st.write("**3. Normalization:**")
            st.write(normalized_data)

            # Step 4: Filtering (using Sastrawi Stopword Remover)
            filtered_data = [stopword_remover.remove(s) for s in normalized_data]
            st.write("**4. Filtering:**")
            st.write(filtered_data)

            # Step 5: Lemmatization (using Sastrawi Stemmer)
            lemmatized_data = [stemmer.stem(s) for s in filtered_data]
            st.write("**5. Lemmatization:**")
            st.write(lemmatized_data)

            # Save preprocessed data to session state for further analysis
            st.session_state['preprocessed_data'] = lemmatized_data
        else:
            st.error("Please enter raw data for preprocessing.")

def dashboard():
    with st.sidebar:
        selected = option_menu(
            menu_title="Navigation",
            options=["Home", "Preprocessing", "Analysis", "Search Product", "Browse CSV", "Logout"],
            icons=["house", "gear", "graph-up", "search", "file-earmark-spreadsheet", "box-arrow-right"],
            menu_icon="cast",
            default_index=0
        )

    if selected == "Home":
        st.title("Home")
        st.write("This application is used for sentiment analysis of Tokopedia product reviews.")
        st.write("**Main Features:**")
        st.write("1. Preprocess raw data into analysis-ready data.")
        st.write("2. Sentiment analysis using the BERT model.")

    elif selected == "Preprocessing":
        preprocessing_page()

    elif selected == "Analysis":
        st.title("Sentiment Analysis")
        st.write("Analyze preprocessed data using the BERT model.")

        if 'preprocessed_data' in st.session_state:
            st.write("**Preprocessed Data:**")
            st.write(st.session_state['preprocessed_data'])

            if st.button("Analyze Sentiments", key="analyze_preprocessed"):
                response = api_predict(st.session_state['preprocessed_data'])
                if response:
                    st.write("**Sentiment Results:**")
                    st.write(f"Positive: {response['Positive']} reviews")
                    st.write(f"Negative: {response['Negative']} reviews")
                else:
                    st.error("Failed to process sentiment analysis.")
        else:
            st.warning("No preprocessed data found. Please go to the Preprocessing page first.")

    elif selected == "Search Product":
        st.title("Search Product")
        product_url = st.text_input("Enter Tokopedia product link:", key="product_url_input")

        if st.button("Search Product", key="search_product_button"):
            if product_url:
                response = api_product_analysis(product_url)
                if response.get("status"):
                    data = response["data"]

                    st.header("Product Overview")
                    st.image(data["image_urls"][0], width=300)
                    st.write(f"**Product Name:** {data['product_name']}")
                    st.write(f"**Description:** {data['product_description']}")
                    st.write(f"**Rating:** {data['rating']} ({data['bintang']} stars)")
                    st.write(f"**Reviews:** {data['ulasan']}")
                    st.write(f"**Shop:** {data['shop_name']}")

                    st.header("Product Sentiment Analysis")
                    st.write(f"**Sentiment Summary:** {data['summary']}")
                    st.write(f"**Positive Reviews:** {data['count_positive']}")
                    st.write(f"**Negative Reviews:** {data['count_negative']}")

                    labels = ['Positive', 'Negative']
                    sizes = [data['count_positive'], data['count_negative']]
                    fig, ax = plt.subplots()
                    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
                    ax.axis('equal')
                    st.pyplot(fig)
                else:
                    st.error("Failed to fetch product data. Please check the link.")

    elif selected == "Browse CSV":
        st.title("Browse CSV for Analysis")
        uploaded_file = st.file_uploader("Upload your CSV file for analysis:", type=["csv"], key="csv_uploader")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("**Uploaded Dataset Preview:**")
            st.write(df.head())

            if "review_text" not in df.columns:
                st.error("The CSV must contain a 'review_text' column.")
            else:
                if st.button("Analyze Sentiments", key="analyze_csv_sentiments"):
                    reviews = df["review_text"].tolist()
                    response = api_predict(reviews)
                    if response:
                        st.write("**Sentiment Results:**")
                        st.write(f"Positive: {response['Positive']} reviews")
                        st.write(f"Negative: {response['Negative']} reviews")
                    else:
                        st.error("Failed to process sentiment analysis.")

    elif selected == "Logout":
        st.session_state['authenticated'] = False
        st.session_state['page'] = "login"
        st.experimental_rerun()

if __name__ == "__main__":
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    if not st.session_state["authenticated"]:
        if st.session_state["page"] == "login":
            login_page()
        elif st.session_state["page"] == "register":
            register_page()
    else:
        dashboard()

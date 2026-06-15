
# ==========================
# IMPORT LIBRARIES
# ==========================

import streamlit as st
import pandas as pd
import pickle
import re 

# ==========================
# PAGE CONFIGURATION
# ==========================

st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="🛒",
    layout="wide"
)


# ==========================
# LOAD TRAINED MODEL
# ==========================

@st.cache_resource
def load_model():

    with open("sentiment_model.pkl", "rb") as file:

        saved = pickle.load(file)

    return (
        saved["vectorizer"],
        saved["model"],
        saved["model_name"]
    )

tfidf, model, model_name = load_model()

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("📌 Project Information")

st.sidebar.markdown(f"""

### 🎯 Objective

Predict customer sentiment from Amazon reviews.

### 🛠 Technologies Used

- Python
- Streamlit
- SpaCy
- Scikit-Learn
- TF-IDF Vectorizer

### 🤖 Model Used

**{model_name}**

### 📊 Sentiment Classes

😊 Positive

😞 Negative

😐 Neutral

""")

# ==========================
# TEXT CLEANING FUNCTION
# ==========================

def data_clean(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    text = " ".join(text.split())

    return text


# ==========================
# TITLE
# ==========================

st.title("🛒 Amazon Review Sentiment Analysis")

st.caption(
    "Machine Learning Based Sentiment Classification System"
)

st.divider()

# ==========================
# WORKFLOW
# ==========================

st.subheader("🔄 Project Workflow")

st.markdown("""

1️⃣ User enters a review.

2️⃣ Text preprocessing is performed.

3️⃣ TF-IDF converts text into numerical features.

4️⃣ Machine Learning model predicts sentiment.

5️⃣ Result is displayed.

""")

st.divider()

# ==========================
# SAMPLE REVIEWS
# ==========================

st.subheader("📝 Try Sample Reviews")

sample = st.selectbox(

    "Choose a sample review",

    (

        "Select",

        "This product is amazing and worth every penny.",

        "Very poor quality and complete waste of money.",

        "The product is okay but delivery was late."

    )

)

# ==========================
# SINGLE REVIEW PREDICTION
# ==========================

st.subheader("✍️ Single Review Prediction")

review = st.text_area(

    "Enter Customer Review",

    value="" if sample == "Select" else sample,

    height=150

)

if st.button("Predict Sentiment"):

    if review.strip() == "":

        st.warning("Please enter a review.")

    else:

        cleaned_review = data_clean(review)

        st.subheader("🧹 Text Preprocessing")

        st.write("Original Review:")

        st.info(review)

        st.write("Cleaned Review:")

        st.success(cleaned_review)

        review_vector = tfidf.transform(

            [cleaned_review]

        )

        prediction = model.predict(

            review_vector

        )[0]

        st.subheader("🎯 Prediction")

        if prediction == "Positive":

            st.success("😊 Positive Review")

        elif prediction == "Negative":

            st.error("😞 Negative Review")

        else:

            st.warning("😐 Neutral Review")

        # Confidence score

        if hasattr(model, "predict_proba"):

            probability = model.predict_proba(

                review_vector

            )[0]

            confidence = max(probability)

            st.subheader("📈 Prediction Confidence")

            st.progress(float(confidence))

            st.write(

                f"Confidence : {confidence*100:.2f}%"

            )

st.divider()

# ==========================
# BULK CSV PREDICTION
# ==========================

st.subheader("📂 Bulk Sentiment Prediction")

uploaded_file = st.file_uploader(

    "Upload CSV File",

    type=["csv"]

)

if uploaded_file:

    try:

        try:

            df = pd.read_csv(

                uploaded_file,

                encoding="utf-8"

            )

        except UnicodeDecodeError:

            uploaded_file.seek(0)

            try:

                df = pd.read_csv(

                    uploaded_file,

                    encoding="cp1252"

                )

            except:

                uploaded_file.seek(0)

                df = pd.read_csv(

                    uploaded_file,

                    encoding="latin1"

                )

        df.columns = df.columns.str.strip()

        if "Review" not in df.columns:

            st.error(

                "CSV must contain a column named 'Review'"

            )

        else:

            df["Cleaned_Review"] = (

                df["Review"]

                .astype(str)

                .apply(data_clean)

            )

            review_vector = tfidf.transform(

                df["Cleaned_Review"]

            )

            df["Predicted_Sentiment"] = (

                model.predict(

                    review_vector

                )

            )

            st.subheader("📊 Dataset Information")

            col1, col2 = st.columns(2)

            col1.metric(

                "Total Reviews",

                len(df)

            )

            col2.metric(

                "Columns",

                len(df.columns)

            )

            st.subheader(

                "📈 Sentiment Distribution"

            )

            st.bar_chart(

                df["Predicted_Sentiment"]

                .value_counts()

            )

            st.subheader(

                "📋 Prediction Results"

            )

            st.dataframe(df)

            csv = (

                df.to_csv(

                    index=False

                )

                .encode("utf-8")

            )

            st.download_button(

                label="📥 Download Results",

                data=csv,

                file_name="sentiment_results.csv",

                mime="text/csv"

            )

    except Exception as e:

        st.error(

            f"Error : {e}"

        )

# ==========================
# FOOTER
# ==========================

st.divider()

st.markdown("""

### 👩‍💻 Developed By

**Group 3**

Machine Learning Project

**Tools Used**

Python | Streamlit | SpaCy | Scikit-Learn | TF-IDF

""")
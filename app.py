import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# --- Load model dan komponen ---
model = joblib.load('RidgeClassifier - Ukulele by Yousician.pkl')
vectorizer = joblib.load('tfidf_vectorizer_Ukulele by Yousician.pkl')
label_encoder = joblib.load('label_encoder_Ukulele by Yousician.pkl')

# --- Judul App ---
st.title("🎵 Sentiment Analysis - Ukulele by Yousician")

# --- Pilih Mode ---
st.header("Pilih Metode Input")
input_mode = st.radio("Mode Input:", ["📝 Input Manual", "📁 Upload CSV"])

# ========================================
# 📌 MODE 1: INPUT MANUAL
# ========================================
if input_mode == "📝 Input Manual":
    st.subheader("Masukkan 1 Review Pengguna")
    
    name = st.text_input("👤 Nama Pengguna:")
    star_rating = st.selectbox("⭐ Bintang Rating:", [1, 2, 3, 4, 5])
    user_review = st.text_area("💬 Review:")
    review_date = st.datetime_input("🗓️ Tanggal & Waktu Submit:", value=datetime.now())
    review_date_str = review_date.strftime("%Y-%m-%d %H:%M")

    if st.button("Prediksi Sentimen"):
        if user_review.strip() == "":
            st.warning("🚨 Silakan isi review terlebih dahulu.")
        else:
            vec = vectorizer.transform([user_review])
            pred = model.predict(vec)
            label = label_encoder.inverse_transform(pred)[0]

            # Buat hasil sebagai DataFrame
            result_df = pd.DataFrame([{
                "name": name if name else "(Anonim)",
                "star_rating": star_rating,
                "date": review_date_str,
                "review": user_review,
                "predicted_sentiment": label
            }])

            st.success("✅ Prediksi berhasil!")
            st.dataframe(result_df)

            # Tombol Download
            csv_manual = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Hasil Manual sebagai CSV",
                data=csv_manual,
                file_name="manual_review_prediction.csv",
                mime="text/csv"
            )

# ========================================
# 📁 MODE 2: UPLOAD CSV
# ========================================
else:
    st.subheader("Upload File CSV Review")
    uploaded_file = st.file_uploader("Pilih file CSV (harus memiliki kolom 'review')", type=['csv'])

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)

            # Validasi kolom
            if 'review' not in df.columns:
                st.error("❌ File harus memiliki kolom 'review'.")
            else:
                # Prediksi
                X_vec = vectorizer.transform(df['review'].fillna(""))
                y_pred = model.predict(X_vec)
                df['predicted_sentiment'] = label_encoder.inverse_transform(y_pred)

                st.success("✅ Prediksi berhasil!")
                st.dataframe(df.head())

                # Download hasil
                csv_result = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Hasil CSV",
                    data=csv_result,
                    file_name="predicted_reviews.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Terjadi error saat membaca file: {e}")

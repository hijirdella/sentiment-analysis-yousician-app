import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import pytz
from datetime import datetime

# === Load model dan komponen ===
model = joblib.load('RidgeClassifier - Ukulele by Yousician.pkl')
vectorizer = joblib.load('tfidf_vectorizer_Ukulele by Yousician.pkl')
label_encoder = joblib.load('label_encoder_Ukulele by Yousician.pkl')

# === Judul Aplikasi ===
st.title("🎵 Sentiment App – Ukulele by Yousician")

# === Pilih Mode Input ===
st.header("🎯 Pilih Metode Input")
input_mode = st.radio("Mode Input:", ["📝 Input Manual", "📁 Upload CSV"])

# ========================================
# 📝 MODE 1: INPUT MANUAL
# ========================================
if input_mode == "📝 Input Manual":
    st.subheader("Masukkan 1 Review Pengguna")

    name = st.text_input("👤 Nama Pengguna:")
    star_rating = st.selectbox("⭐ Rating Bintang:", [1, 2, 3, 4, 5])
    review_text = st.text_area("💬 Review Pengguna:")

    # Default waktu Waktu Indonesia Barat
    wib = pytz.timezone("Asia/Jakarta")
    now = datetime.now(wib)

    review_day = st.date_input("📅 Tanggal Review:", value=now.date())
    review_time = st.time_input("⏰ Waktu Review:", value=now.time())

    review_datetime = datetime.combine(review_day, review_time)
    review_datetime_wib = wib.localize(review_datetime)
    review_date_str = review_datetime_wib.strftime("%Y-%m-%d %H:%M")

    if st.button("🔍 Prediksi Sentimen"):
        if review_text.strip() == "":
            st.warning("⚠️ Silakan isi review terlebih dahulu.")
        else:
            X = vectorizer.transform([review_text])
            y_pred = model.predict(X)
            label = label_encoder.inverse_transform(y_pred)[0]

            hasil_df = pd.DataFrame([{
                'name': name if name else "(Anonim)",
                'star_rating': star_rating,
                'date': review_date_str,
                'review': review_text,
                'predicted_sentiment': label
            }])

            st.success("✅ Prediksi berhasil!")
            st.dataframe(hasil_df)

            # Tombol Download
            csv_download = hasil_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Hasil CSV",
                data=csv_download,
                file_name="hasil_prediksi_manual.csv",
                mime="text/csv"
            )

# ========================================
# 📁 MODE 2: UPLOAD CSV
# ========================================
else:
    st.subheader("📤 Upload File CSV Review")
    file = st.file_uploader("Unggah file CSV dengan kolom 'review'", type=["csv"])

    if file:
        try:
            df = pd.read_csv(file)

            if 'review' not in df.columns:
                st.error("❌ File harus memiliki kolom 'review'.")
            else:
                # Prediksi sentimen
                X = vectorizer.transform(df['review'].fillna(""))
                y_pred = model.predict(X)
                df['predicted_sentiment'] = label_encoder.inverse_transform(y_pred)

                # Label Indonesia
                df['sentimen_indonesia'] = df['predicted_sentiment'].map({'positive': 'Positif', 'negative': 'Negatif'})

                # Konversi kolom tanggal jika tersedia
                if 'date' in df.columns:
                    try:
                        df['date'] = pd.to_datetime(df['date'], errors='coerce')
                        min_date = df['date'].min().date()
                        max_date = df['date'].max().date()
                        selected_range = st.date_input("📅 Filter Tanggal Review", [min_date, max_date])
                        if len(selected_range) == 2:
                            df = df[(df['date'].dt.date >= selected_range[0]) & (df['date'].dt.date <= selected_range[1])]
                    except:
                        st.warning("📌 Kolom 'date' tidak valid sebagai tanggal.")

                # Filter sentimen
                selected_sentiment = st.multiselect("🎭 Filter Sentimen", ['Positif', 'Negatif'], default=['Positif', 'Negatif'])
                df = df[df['sentimen_indonesia'].isin(selected_sentiment)]

                # === Visualisasi Distribusi Sentimen ===
                st.markdown("### 📊 Distribusi Sentimen")

                sentiment_counts = df['sentimen_indonesia'].value_counts().rename_axis('Sentimen').reset_index(name='Jumlah')

                col1, col2 = st.columns(2)

                with col1:
                    fig1, ax1 = plt.subplots()
                    colors = sentiment_counts['Sentimen'].map({'Positif': 'blue', 'Negatif': 'red'})
                    bars = ax1.bar(sentiment_counts['Sentimen'], sentiment_counts['Jumlah'], color=colors)

                    # Tambahkan label jumlah di atas bar
                    for bar in bars:
                        height = bar.get_height()
                        ax1.annotate(f'{height}', xy=(bar.get_x() + bar.get_width() / 2, height),
                                     xytext=(0, 5), textcoords="offset points", ha='center', va='bottom', fontsize=10)

                    ax1.set_ylabel("Jumlah Review")
                    ax1.set_xlabel("Sentimen")
                    ax1.set_title("Distribusi Sentimen (Bar)")
                    st.pyplot(fig1)

                with col2:
                    fig2, ax2 = plt.subplots()
                    pie_colors = ['blue' if s == 'Positif' else 'red' for s in sentiment_counts['Sentimen']]
                    ax2.pie(sentiment_counts['Jumlah'], labels=sentiment_counts['Sentimen'],
                            autopct='%1.1f%%', colors=pie_colors, textprops={'fontsize': 10})
                    ax2.set_title("Distribusi Sentimen (Pie)")
                    st.pyplot(fig2)

                # === Tampilkan Tabel Data Asli ===
                st.markdown("### 📄 Data Review")
                st.dataframe(df.drop(columns=["predicted_sentiment", "sentimen_indonesia"]))

                # Download CSV
                csv_result = df.drop(columns=["sentimen_indonesia"]).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Hasil Prediksi CSV",
                    data=csv_result,
                    file_name="hasil_prediksi.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat memproses file: {e}")

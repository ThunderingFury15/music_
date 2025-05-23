import os
import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pandas as pd
import soundfile as sf

st.set_page_config(page_title="🎵 Music App", layout="centered")

# Sidebar chọn chức năng
page = st.sidebar.radio("📂 Chọn chức năng", ["🎧 Nghe nhạc", "🔍 Dự đoán thể loại"])

# -------------------------------------
# 🎧 Trang 1: Nghe nhạc từ thư mục local
# -------------------------------------
if page == "🎧 Nghe nhạc":
    st.title("🎧 Trình phát nhạc đơn giản")

    song_dir = "assets/songs"
    if not os.path.exists(song_dir):
        st.error(f"Thư mục `{song_dir}` không tồn tại!")
        st.stop()

    songs = [f for f in os.listdir(song_dir) if f.endswith(".mp3")]
    if not songs:
        st.warning("Không có file .mp3 nào trong thư mục.")
        st.stop()

    song_names = [os.path.splitext(song)[0] for song in songs]
    song_map = dict(zip(song_names, songs))

    selected_song_name = st.selectbox("🎵 Chọn bài hát", song_names)
    selected_song_path = os.path.join(song_dir, song_map[selected_song_name])

    st.markdown(f"### ▶️ Đang phát: **{selected_song_name}**")
    with open(selected_song_path, "rb") as f:
        st.audio(f.read(), format="audio/mp3")

# -------------------------------------
# 🔍 Trang 2: Dự đoán thể loại nhạc
# -------------------------------------
elif page == "🔍 Dự đoán thể loại":
    st.title("🔍 Dự đoán thể loại nhạc")

    # Load mô hình
    model_path = "model/genre_classifier.pkl"
    if not os.path.exists(model_path):
        st.error("Không tìm thấy mô hình tại 'model/genre_classifier.pkl'")
        st.stop()
    model = joblib.load(model_path)

    genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
              'jazz', 'metal', 'pop', 'reggae', 'rock']
    genre_emojis = {
        'blues': '🎼', 'classical': '🎻', 'country': '🤠', 'disco': '🪩',
        'hiphop': '🎤', 'jazz': '🎷', 'metal': '🤘', 'pop': '🎧',
        'reggae': '🟢', 'rock': '🎸'
    }

    st.markdown("📤 **Tải lên file `.wav` (30 giây)** để dự đoán thể loại.")
    file = st.file_uploader("Chọn file", type=["wav"])

    if file:
        with st.spinner("Đang phân tích âm thanh..."):
            st.audio(file, format="audio/wav")

            # Load audio với soundfile
            y, sr = sf.read(file)
            if y.ndim > 1:
                y = y.mean(axis=1)
            y = y[:sr * 30]  # lấy 30 giây

            # Trích xuất đặc trưng
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)

            # Ước lượng tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_val = float(tempo) if np.isscalar(tempo) else float(tempo[0])
            st.markdown(f"**🎚 Tempo (BPM):** `{tempo_val:.2f}`")

            # Dự đoán thể loại
            probs = model.predict_proba(mfcc_mean)[0]
            prediction = model.predict(mfcc_mean)[0]
            confidence = np.max(probs) * 100
            emoji = genre_emojis.get(prediction, '')

            st.markdown(f"### 🎯 Thể loại dự đoán: **{prediction.upper()}** {emoji}")
            st.markdown(f"**📈 Độ tin cậy:** {confidence:.2f}%")

            # Hiển thị top 3
            top3_idx = np.argsort(probs)[::-1][:3]
            df_top3 = pd.DataFrame({
                "Thể loại": [genres[i].capitalize() for i in top3_idx],
                "Xác suất (%)": [probs[i]*100 for i in top3_idx]
            })
            st.markdown("### 📊 Xác suất top 3 thể loại")
            st.bar_chart(df_top3.set_index("Thể loại"))

            # Sóng âm
            with st.expander("📈 Xem sóng âm"):
                fig, ax = plt.subplots()
                librosa.display.waveshow(y, sr=sr)
                ax.set_title("Waveform")
                st.pyplot(fig)

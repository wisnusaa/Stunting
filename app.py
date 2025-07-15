import streamlit as st
import joblib
import numpy as np

# Fungsi untuk memuat model dan scaler yang sudah disimpan
# @st.cache_resource digunakan untuk caching, sehingga model tidak perlu di-load ulang setiap kali ada interaksi pengguna
@st.cache_resource
def load_model_and_scaler():
    """Memuat model dan scaler dari file joblib."""
    model = joblib.load('stunting_model.joblib')
    scaler = joblib.load('scaler.joblib')
    return model, scaler

# Memuat model dan scaler
try:
    model, scaler = load_model_and_scaler()
except FileNotFoundError:
    st.error("File model atau scaler tidak ditemukan. Pastikan 'stunting_model.joblib' dan 'scaler.joblib' ada di direktori yang sama.")
    st.stop()


# Judul dan deskripsi aplikasi
st.set_page_config(page_title="Prediksi Stunting", page_icon="👶")
st.title('👶 Aplikasi Prediksi Stunting Anak')
st.write(
    "Aplikasi ini menggunakan model Machine Learning untuk memprediksi risiko stunting pada anak "
    "berdasarkan data umur, jenis kelamin, berat, dan tinggi badan. Masukkan data di bawah ini untuk mendapatkan prediksi."
)

# Membuat form untuk input pengguna
with st.form("prediction_form"):
    st.header("Masukkan Data Anak:")
    
    # Input Umur (bulan)
    umur = st.number_input('Umur (dalam bulan)', min_value=0, max_value=72, value=12, help="Masukkan umur anak antara 0 hingga 72 bulan.")
    
    # Input Jenis Kelamin
    # Mapping dari teks ke nilai numerik sesuai dengan training data
    jenis_kelamin_map = {'Laki-laki': 0, 'Perempuan': 1}
    jenis_kelamin_text = st.selectbox('Jenis Kelamin', options=['Laki-laki', 'Perempuan'], help="Pilih jenis kelamin anak.")
    jenis_kelamin = jenis_kelamin_map[jenis_kelamin_text]
    
    # Input Berat Badan (kg)
    berat = st.number_input('Berat Badan (kg)', min_value=1.0, max_value=30.0, value=8.0, format="%.2f", help="Masukkan berat badan anak dalam kilogram.")
    
    # Input Tinggi Badan (cm)
    tinggi = st.number_input('Tinggi Badan (cm)', min_value=40.0, max_value=120.0, value=70.0, format="%.2f", help="Masukkan tinggi badan anak dalam sentimeter.")
    
    # Tombol submit
    submit_button = st.form_submit_button(label='Prediksi Stunting')

# Logika setelah form disubmit
# Logika setelah form disubmit
if submit_button:
    # --- PERUBAHAN DIMULAI DI SINI ---

    # 1. Pisahkan fitur kontinu dan kategori
    # Fitur kontinu adalah yang akan kita scale
    continuous_features = np.array([[umur, berat, tinggi]])
    
    # Fitur kategori (Jenis Kelamin) tidak di-scale
    # Nilainya sudah 0 atau 1
    categorical_feature = jenis_kelamin

    # 2. Lakukan scaling HANYA pada fitur kontinu
    # Scaler sekarang menerima input dengan 3 fitur, sesuai dengan saat ia dilatih
    scaled_continuous_features = scaler.transform(continuous_features)

    # 3. Gabungkan kembali semua fitur dalam urutan yang BENAR untuk model
    # Model Anda dilatih dengan urutan: [Umur, Jenis Kelamin, Berat, Tinggi]
    # Jadi, kita harus menyusun kembali array input sesuai urutan tersebut
    input_data = np.array([[
        scaled_continuous_features[0,0], # Umur yang sudah di-scale
        categorical_feature,             # Jenis Kelamin (tidak di-scale)
        scaled_continuous_features[0,1], # Berat yang sudah di-scale
        scaled_continuous_features[0,2]  # Tinggi yang sudah di-scale
    ]])
    
    # --- AKHIR PERUBAHAN ---

    # 4. Melakukan prediksi dengan data yang sudah siap
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)
    
    st.header("Hasil Prediksi:")
    
    # Menampilkan hasil prediksi
    # Berdasarkan notebook, 1 = Stunting, 0 = Tidak Stunting
    if prediction[0] == 1:
        st.error('**Berisiko Stunting**')
        st.write(f"Model memprediksi dengan probabilitas **{prediction_proba[0][1]*100:.2f}%** bahwa anak berisiko mengalami stunting.")
    else:
        st.success('**Tidak Berisiko Stunting**')
        st.write(f"Model memprediksi dengan probabilitas **{prediction_proba[0][0]*100:.2f}%** bahwa anak tidak berisiko mengalami stunting.")

    st.info("Catatan: Prediksi ini berdasarkan model Machine Learning dan bukan merupakan diagnosis medis. Selalu konsultasikan dengan tenaga kesehatan profesional.")

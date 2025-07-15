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
if submit_button:
    # 1. Menyiapkan data input untuk model
    # Urutan fitur harus sama persis dengan saat training: [Umur (bulan), Jenis Kelamin, Berat Badan (kg), Tinggi Badan (cm)]
    input_data = np.array([[umur, jenis_kelamin, berat, tinggi]])
    
    # 2. Melakukan scaling pada input data
    # Kita hanya transform, tidak fit_transform, karena kita menggunakan skala dari data training
    scaled_input_data = scaler.transform(input_data)
    
    # 3. Melakukan prediksi
    prediction = model.predict(scaled_input_data)
    prediction_proba = model.predict_proba(scaled_input_data)
    
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
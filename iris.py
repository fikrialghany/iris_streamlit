# Aplikasi Streamlit untuk Prediksi Bunga Iris dengan Naive Bayes
# --------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from PIL import Image
import base64
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Prediksi Spesies Bunga Iris",
    page_icon="🌸",
    layout="wide"
)

# Fungsi untuk memuat gambar dari URL atau path file
def load_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# URL-URL gambar bunga Iris
# Catatan: Dalam aplikasi nyata, simpan gambar ini di folder "images" di direktori aplikasi Anda
# dan gunakan path seperti "images/iris_setosa.jpg"
IMAGES = {
    "Iris-setosa": "https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg",
    "Iris-versicolor": "https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg",
    "Iris-virginica": "https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg"
}

# Fungsi untuk memuat model dari file pickle
@st.cache_resource
def load_model():
        with open('naive_bayes_iris_model.pkl', 'rb') as file:
            model_data = pickle.load(file)
        st.sidebar.success("✅ Model berhasil dimuat dari file!")
        return model_data

# Fungsi untuk prediksi
def predict_species(model_data, features):
    # Mengubah features menjadi array numpy
    input_data = np.array([features])
    
    # Scaling
    input_data_scaled = model_data['scaler'].transform(input_data)
    
    # Prediksi
    prediction = model_data['model'].predict(input_data_scaled)[0]
    
    # Probabilitas
    probabilities = model_data['model'].predict_proba(input_data_scaled)[0]
    prob_dict = {class_name: prob for class_name, prob in zip(model_data['classes'], probabilities)}
    
    return prediction, prob_dict

# Fungsi untuk menampilkan informasi spesies
def display_species_info(species):
    if species == "Iris-setosa":
        st.write("""
        ### Iris Setosa
        **Iris setosa** adalah spesies bunga iris yang berasal dari timur laut Asia, termasuk Jepang, Rusia, dan Alaska. 
        Memiliki ciri khas kelopak biru-ungu dengan kelopak yang lebih kecil dibandingkan spesies iris lainnya. 
        Tanaman ini tumbuh dengan baik di daerah lembab dan tanah yang kaya.
        """)
    elif species == "Iris-versicolor":
        st.write("""
        ### Iris Versicolor
        **Iris versicolor**, juga dikenal sebagai Iris Biru Besar atau Iris Biru Bendera, adalah tanaman asli Amerika Utara.
        Spesies ini memiliki kelopak berwarna ungu ke biru dengan pola veining kuning. 
        Biasanya tumbuh di area berawa, tepi danau, dan padang rumput lembab.
        """)
    elif species == "Iris-virginica":
        st.write("""
        ### Iris Virginica
        **Iris virginica** adalah spesies iris asli Amerika Utara yang tumbuh di daerah basah seperti rawa-rawa dan tepi sungai.
        Bunga ini memiliki kelopak berwarna ungu-biru yang lebih besar dari Iris setosa dan versicolor.
        Spesies ini juga dikenal sebagai Iris Virginia atau Iris Biru Air.
        """)

# Fungsi untuk plot distribusi fitur
@st.cache_data
def plot_feature_distribution():
    df = pd.read_csv('Iris.csv')
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
    
    for i, feature in enumerate(features):
        for species in df['Species'].unique():
            sns.kdeplot(df[df['Species'] == species][feature], ax=axes[i], label=species)
        axes[i].set_title(f'Distribusi {feature}')
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel('Densitas')
        axes[i].legend()
    
    plt.tight_layout()
    return fig

# Fungsi untuk menampilkan nilai prediksi pada gauge
def create_gauge_chart(probability, species):
    # Membuat gauge chart sederhana
    fig, ax = plt.subplots(figsize=(3, 3))
    
    # Warna berdasarkan spesies
    colors = {
        "Iris-setosa": "lightcoral",
        "Iris-versicolor": "lightblue",
        "Iris-virginica": "lightgreen"
    }
    
    # Membuat chart
    ax.pie([probability, 1-probability], colors=[colors[species], 'white'], 
           startangle=90, counterclock=False,
           wedgeprops={'width': 0.4, 'edgecolor': 'white'})
    
    # Menambahkan teks di tengah
    ax.text(0, 0, f"{probability:.2%}", ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Menambahkan judul
    ax.set_title(f"Probabilitas\n{species.replace('Iris-', '')}", fontsize=12)
    
    # Menghilangkan axis
    ax.axis('equal')
    ax.set_frame_on(False)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    return fig

# Header aplikasi
st.title("🌸 Prediksi Spesies Bunga Iris")
st.markdown("""
Aplikasi ini memprediksi spesies bunga Iris berdasarkan dimensi sepal dan petal
menggunakan algoritma Naive Bayes. Gunakan slider di sidebar untuk mengatur nilai input.
""")

# Sidebar untuk train model
st.sidebar.header("Model Management")


# Sidebar untuk input
st.sidebar.header("Input Parameter")
st.sidebar.markdown("Gunakan slider di bawah untuk menyesuaikan dimensi bunga Iris:")

# Slider untuk input
sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0, 0.1)
petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 3.8, 0.1)
petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.2, 0.1)

# Input features
features = [sepal_length, sepal_width, petal_length, petal_width]

# Memuat model
model_data = load_model()

# Prediksi
prediction, probabilities = predict_species(model_data, features)

# Tampilkan hasil dalam dua kolom
col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("Hasil Prediksi")
    # Menentukan warna berdasarkan prediksi
    color_map = {
        'Iris-setosa': '#FF6B6B', 
        'Iris-versicolor': '#4D96FF', 
        'Iris-virginica': '#6BCB77'
    }
    color = color_map[prediction]
    
    st.markdown(f"""
    <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; text-align:center;">
        <h2 style="color:{color};">
            {prediction.replace('Iris-', '')}
        </h2>
        <p>Probabilitas: {probabilities[prediction]:.2%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Menampilkan informasi spesies
    display_species_info(prediction)
    
    # Tampilkan gauge charts untuk setiap probabilitas
    st.subheader("Probabilitas per Kelas")
    gauge_cols = st.columns(3)
    for i, (species, prob) in enumerate(probabilities.items()):
        with gauge_cols[i]:
            gauge_fig = create_gauge_chart(prob, species)
            st.pyplot(gauge_fig)

with col2:
    st.subheader("Gambar Bunga")
    try:
        # Menampilkan gambar berdasarkan prediksi
        st.image(IMAGES[prediction], caption=f"Bunga {prediction}", width=400)
    except:
        st.error("Tidak dapat menampilkan gambar. Periksa koneksi internet Anda.")
    
    # Menampilkan detail input
    st.subheader("Detail Parameter Input")
    
    # Tampilan tabel parameter
    input_df = pd.DataFrame({
        'Parameter': ['Sepal Length (cm)', 'Sepal Width (cm)', 'Petal Length (cm)', 'Petal Width (cm)'],
        'Nilai': features
    })
    st.table(input_df)
    
    # Menampilkan posisi input pada distribusi data
    st.subheader("Posisi Input pada Distribusi")
    st.pyplot(plot_feature_distribution())

# Footer
st.markdown("""
---
Dibuat sebagai demo untuk klasifikasi Bunga Iris menggunakan Naive Bayes dan Streamlit.
""")

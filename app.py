import streamlit as st
import base64
import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px

# ======================= #
# Konfigurasi Streamlit   #
# ======================= #
st.set_page_config(page_title="Analisis Kinerja Prabowo", layout="wide")

def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
    """, unsafe_allow_html=True)

add_bg_from_local("image/Save-Raja-Ampat.jpg")

st.markdown("""<h1 style='text-align: center; color: white;'>
    Analisis Sentimen Komentar TikTok terhadap Kinerja Presiden Prabowo<br> dalam Penanganan Tambang di Raja Ampat
</h1>""", unsafe_allow_html=True)

# ======================= #
# Preprocessing dan Label #
# ======================= #
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

positif_phrases = [
    'bagus', 'baik', 'hebat', 'mantap', 'top', 'sukses', 'kerja bagus',
    'puas', 'terbaik', 'bagus banget', 'luar biasa', 'dukung presiden',
    'jangan salahkan presiden', 'gapernah nyesel', 'tetap semangat',
    'aku percaya bapak', 'terima kasih pak', 'kami mendukungmu',
    'presiden terbaik', 'keren banget', 'lanjutkan pak', 'aku bangga',
    'mana nih', 'makasih', 'ga salah pilih', 'mana yg bilang', 'di jaga ketikannya',
    'pak', 'aku dukung', 'kita dukung', 'percaya', 'aku percaya',
    'terimakasih', 'tetap dukung', 'saya yakin', 'harap pak', 'selamatkan alam',
    'berharap', 'dukung bapak',

    'selamatkan raja ampat', 'jaga lingkungan', 'lestarikan alam',
    'tolong jaga hutan', 'selamatkan ekosistem', 'dukung lingkungan',
    'dukung pelestarian', 'presiden peduli lingkungan', 'stop tambang ilegal',
    'hentikan perusakan', 'cegah kerusakan', 'selamatkan hutan',
    'peduli lingkungan', 'jaga raja ampat', 'selamatkan tambrauw',

     'presiden kerja', 'bagus kerjanya', 'tanggapan bagus', 'tanggap cepat',
    'peduli lingkungan', 'kinerja bagus', 'bantu lingkungan'
    


]

negatif_phrases = [
    'buruk', 'jelek', 'gagal', 'bohong', 'parah', 'salah pilih',
    'salah presiden', 'kacau', 'tidak puas', 'mengecewakan',
    'salah besar', 'salah milih', 'hancur', 'nggak becus',
    'day 1 salah', 'prabowo manis di awal', 'bener2 nyesel', 'kecewa',
    'presiden gagal', 'bukan presiden gue', 'pilihan salah', 'ganti presiden',
    'berhenti', 'bukan presiden', 'udah salah', 'salah total', 'tolong stop',
    'jangan lanjutkan', 'hancurkan', 'kerusakan', 'dikhianati',
    'rusak alam', 'janji palsu', 'sama aja bohong',

    'merusak raja ampat', 'rusak raja ampat', 'kerusakan lingkungan',
    'penghancur alam', 'pengkhianat lingkungan', 'tambang ilegal',
    'kerusakan tambah parah', 'rusak hutan', 'penambangan liar',
    'lingkungan rusak', 'tidak peduli raja ampat', 'ekosistem rusak',
    'prabowo rusak alam', 'izin tambang sembarangan', 'jual raja ampat',
    'penambangan brutal', 'hancurkan raja ampat', 'prabowo perusak alam',   'tidak pantas', 'tidak layak', 'parah banget', 'bencana alam', 'tidak tanggap','pengrusakan'

    
]


def label_sentimen(text):
    for phrase in positif_phrases:
        if phrase in text:
            return 'baik'
    for phrase in negatif_phrases:
        if phrase in text:
            return 'buruk'
    return 'unknown'

# ======================= #
# Sidebar Upload          #
# ======================= #
st.sidebar.header("\U0001F50D Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV komentar", type=["csv"])

# ======================= #
# Hybrid Predict Function #
# ======================= #
def hybrid_predict(comment, model, vectorizer):
    clean = clean_text(comment)
    for phrase in negatif_phrases:
        if phrase in clean:
            return 0, 1.0
    for phrase in positif_phrases:
        if phrase in clean:
            return 1, 1.0
    vec = vectorizer.transform([clean])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0][pred]
    return pred, prob

# ======================= #
# Proses Klasifikasi      #
# ======================= #
if uploaded_file:
        df_raw = pd.read_csv(uploaded_file)
    df_raw['komentar'] = df_raw['komentar'].astype(str)
    df_raw['komentar_clean'] = df_raw['komentar'].apply(clean_text)

    df_raw['is_relevant'] = df_raw['komentar_clean'].apply(lambda x: any(k in x for k in filter_keywords))
    df_relevan = df_raw[df_raw['is_relevant'] == True].copy()
    df_relevan['label'] = df_relevan['komentar_clean'].apply(label_sentimen)

    df_labeled = df_relevan[df_relevan['label'].isin(['baik', 'buruk'])].copy()
    df_labeled['label_binary'] = df_labeled['label'].map({'baik': 1, 'buruk': 0})

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_labeled['komentar_clean'])
    y = df_labeled['label_binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    dt = DecisionTreeClassifier(random_state=42)
    rf.fit(X_train, y_train)
    dt.fit(X_train, y_train)

    # Prediksi komentar relevan tapi belum terklasifikasi
    df_unknown = df_relevan[df_relevan['label'] == 'unknown'].copy()
    X_unknown = vectorizer.transform(df_unknown['komentar_clean'])
    df_unknown['label_binary'] = rf.predict(X_unknown)
    df_unknown['label'] = df_unknown['label_binary'].map({1: 'baik', 0: 'buruk'})

    df_final = pd.concat([df_labeled, df_unknown], ignore_index=True)

    st.session_state.update({
        'df': df_final,
        'vectorizer': vectorizer,
        'X_test': X_test,
        'y_test': y_test,
        'model_rf': rf,
        'model_dt': dt,
        'total_dataset': len(df_raw),
        'total_relevan': len(df_relevan),
        'total_terklasifikasi': len(df_final)
    })

 

# ======================= #
# Tabs UI                 #
# ======================= #
tab1, tab2, tab3, tab4 = st.tabs(["\U0001F4C4 Dataset", "\U0001F4CA Evaluasi", "\U0001F4AC Prediksi", "\u2601\ufe0f Visualisasi"])

# Tab 1: Dataset
with tab1:
    st.subheader("📄 Ringkasan Dataset Komentar")
    if 'df' in st.session_state:
        df = st.session_state['df']
        total_all = st.session_state['total_dataset']
        total_relevan = st.session_state['total_relevan']
        total_klasifikasi = st.session_state['total_terklasifikasi']
        total_baik = (df['label_binary'] == 1).sum()
        total_buruk = (df['label_binary'] == 0).sum()

        st.dataframe(df[['komentar', 'komentar_clean', 'label']], use_container_width=True)

        col_top1, col_top2 = st.columns(2)
        with col_top1:
            st.metric("Total Komentar Awal", total_all)
        with col_top2:
            st.metric("Komentar Relevan (Raja Ampat)", total_relevan)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Terklasifikasi", total_klasifikasi)
        with col2:
            st.metric("Baik ✅", f"{total_baik} ({total_baik/total_klasifikasi*100:.2f}%)")
        with col3:
            st.metric("Buruk ❌", f"{total_buruk} ({total_buruk/total_klasifikasi*100:.2f}%)")

        st.plotly_chart(px.pie(names=["Baik", "Buruk"], values=[total_baik, total_buruk], title="Distribusi Sentimen Komentar"))

# Tab 2: Evaluasi
with tab2:
    def show_evaluation(model, name):
        y_test = st.session_state['y_test']
        X_test = st.session_state['X_test']
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        st.markdown(f"### Evaluasi: {name}")
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title("Confusion Matrix")
            st.pyplot(fig)
        with col2:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            ax.plot([0, 1], [0, 1], linestyle='--')
            ax.set_title("ROC Curve")
            st.pyplot(fig)
        st.text(classification_report(y_test, y_pred))

    if 'model_rf' in st.session_state:
        show_evaluation(st.session_state['model_rf'], "Random Forest")
        show_evaluation(st.session_state['model_dt'], "Decision Tree")

# Tab 3: Prediksi
with tab3:
    st.subheader("💬 Prediksi Komentar Baru")
    user_input = st.text_area("Masukkan komentar TikTok...")
    if st.button("Klasifikasikan"):
        if 'model_rf' in st.session_state:
            pred, prob = hybrid_predict(user_input, st.session_state['model_rf'], st.session_state['vectorizer'])
            label = "BAIK ✅" if pred else "BURUK ❌"

            st.markdown("**Komentar Asli:**")
            st.success(user_input)

            st.markdown(f"### Hasil: {label}")
            st.markdown(f"Confidence: `{prob:.2f}`")
        else:
            st.warning("Model belum tersedia. Upload & klasifikasikan dataset terlebih dahulu.")

# Tab 4: WordCloud & Bigram
with tab4:
    st.subheader("☁️ WordCloud & Bigram")
    if 'df' in st.session_state:
        df = st.session_state['df']
        baik = " ".join(df[df['label'] == 'baik']['komentar_clean'])
        buruk = " ".join(df[df['label'] == 'buruk']['komentar_clean'])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Komentar Baik**")
            wc = WordCloud(width=400, height=300, background_color='white').generate(baik)
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

        with col2:
            st.markdown("**Komentar Buruk**")
            wc = WordCloud(width=400, height=300, background_color='white', colormap='Reds').generate(buruk)
            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")
            st.pyplot(fig)

        st.markdown("**Top Bigram**")
        bigram = CountVectorizer(ngram_range=(2, 2), stop_words='english')
        X2 = bigram.fit_transform(df['komentar_clean'])
        total = X2.sum(axis=0)
        bigram_freq = [(word, total[0, idx]) for word, idx in bigram.vocabulary_.items()]
        top_bigrams = sorted(bigram_freq, key=lambda x: x[1], reverse=True)[:10]
        df_bigram = pd.DataFrame(top_bigrams, columns=["Bigram", "Frekuensi"])
        st.bar_chart(df_bigram.set_index("Bigram"))

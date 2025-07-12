import streamlit as st
from streamlit_option_menu import option_menu

# Konfigurasi minimal
st.set_page_config(
    page_title="Supply Chain Optimizer",
    page_icon="⚙️",
    layout="wide"
)

# Hanya satu menu navigasi
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Optimasi Rantai Pasok"],
        icons=["house", "gear"],
        default_index=1,
        styles={
            "container": {"padding": "5px"},
            "icon": {"color": "#FF4B4B", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin": "3px"},
        }
    )

if selected == "Home":
    st.title("🚀 Industrial Supply Chain Optimizer")
    st.markdown("""
    ## Aplikasi Matematika Model Industri
    **Fokus Utama:** Optimasi Rantai Pasok dengan Algoritma Genetika
    - Minimalisasi biaya produksi & transportasi
    - Alokasi sumber daya optimal
    - Kendala emisi karbon
    - Visualisasi jaringan interaktif
    """)
    
    st.image("https://i.imgur.com/8Qb7Jqg.png", caption="Diagram Optimasi Rantai Pasok")

else:
    from pages import Optimasi_Rantai_Pasok
    Optimasi_Rantai_Pasok.main()
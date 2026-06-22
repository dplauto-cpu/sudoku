"""
app.py — Sudoku Solver con Streamlit
Sube una foto de un sudoku y obtén la solución.

Para ejecutar:
    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import sys

# ── Configuración de la página ───────────────────────────────────────────
st.set_page_config(
    page_title="Sudoku Solver",
    page_icon="🧩",
    layout="wide"
)

# ── Rutas de los modelos ─────────────────────────────────────────────────
# Ajusta estas rutas si ejecutas en local
DIR_APP     = os.path.dirname(os.path.abspath(__file__))
DIR_MODELOS = os.path.join(DIR_APP, "MODELOS")
RUTA_YOLO   = os.path.join(DIR_MODELOS, "modelo_yolo.pt")
RUTA_CNN    = os.path.join(DIR_MODELOS, "modelo_digitos_v3.keras")
RUTA_SOLVER = os.path.join(DIR_MODELOS, "modelo_solver_1m.keras")

# ── Cargar modelos (solo una vez gracias a cache) ────────────────────────
@st.cache_resource
def cargar_modelos():
    """
    @st.cache_resource guarda los modelos en memoria entre ejecuciones.
    Sin esto, los modelos se recargarían cada vez que el usuario sube una foto.
    """
    sys.path.append(DIR_APP)
    from pipeline import cargar_modelos as _cargar
    ruta_solver = RUTA_SOLVER if os.path.exists(RUTA_SOLVER) else None
    return _cargar(RUTA_YOLO, RUTA_CNN, ruta_solver)

# ── Interfaz principal ───────────────────────────────────────────────────
st.title("🧩 Sudoku para gente vaga como tú")
st.markdown("Sube una foto de un sudoku y ya veremos lo que pasa.")
st.markdown("---")

# Comprobar que los modelos existen antes de cargar
if not os.path.exists(RUTA_YOLO) or not os.path.exists(RUTA_CNN):
    st.error("❌ No se encuentran los modelos. "
             "Asegúrate de que modelo_yolo.pt y modelo_digitos_v3.keras "
             "están en la carpeta MODELOS/ junto a app.py.")
    st.stop()

# Cargar modelos
with st.spinner("Cargando modelos... (solo la primera vez)"):
    modelo_yolo, modelo_cnn, modelo_solver = cargar_modelos()

# ── Subir imagen ─────────────────────────────────────────────────────────
foto = st.file_uploader(
    "Sube una foto del sudoku",
    type=["jpg", "jpeg", "png"],
    help="La foto debe mostrar la cuadrícula completa y estar bien iluminada."
)

if foto is not None:
    # Guardar la imagen subida en un archivo temporal
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(foto.read())
        ruta_tmp = tmp.name

    # Mostrar imagen original
    img_original = Image.open(ruta_tmp)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Foto original")
        st.image(img_original, width="stretch")

    # Procesar
    with st.spinner("Analizando el sudoku..."):
        sys.path.append(DIR_APP)
        from pipeline import mostrar_resultado

        overlay, matriz, solucion, mensaje = mostrar_resultado(
            ruta_tmp, modelo_yolo, modelo_cnn, modelo_solver
        )

    # Mostrar resultado
    with col2:
        st.subheader("Solución")

        if overlay is not None:
            # Convertir BGR→RGB para Streamlit
            overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB) if len(overlay.shape) == 3 else overlay
            st.image(overlay_rgb, width="stretch")
            st.success(mensaje)
        else:
            st.image(img_original, width="stretch")
            st.error(mensaje)

    # ── Mostrar matrices ──────────────────────────────────────────────────
    if matriz is not None and solucion is not None:
        st.markdown("---")
        col3, col4 = st.columns(2)

        with col3:
            st.subheader("Dígitos detectados")
            tabla_detectada = ""
            for i, fila in enumerate(matriz):
                if i > 0 and i % 3 == 0:
                    tabla_detectada += "\n"
                fila_str = "  ".join(str(int(n)) if n != 0 else "·" for n in fila)
                tabla_detectada += fila_str + "\n"
            st.code(tabla_detectada)

        with col4:
            st.subheader("Solución completa")
            tabla_solucion = ""
            for i, fila in enumerate(solucion):
                if i > 0 and i % 3 == 0:
                    tabla_solucion += "\n"
                fila_str = "  ".join(str(int(n)) for n in fila)
                tabla_solucion += fila_str + "\n"
            st.code(tabla_solucion)

    # Limpiar archivo temporal
    os.unlink(ruta_tmp)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "Proyecto Deep Learning · YOLO + CNN + Backtracking · Junio 2026",
    help="Stack: Ultralytics YOLO, TensorFlow/Keras, OpenCV, Streamlit"
)

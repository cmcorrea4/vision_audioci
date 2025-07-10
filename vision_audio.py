import streamlit as st
import pandas as pd
import openai
from openai import OpenAI
import base64
import io
import json
from PIL import Image
import numpy as np
from typing import List, Dict, Any
import re
from datetime import datetime

# Configuración de la página
st.set_page_config(
    page_title="Identificador de Productos de Madera IA",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🪵 Identificador de Productos de Madera con IA")
st.markdown("*Analiza imágenes y audio para identificar productos de madera usando OpenAI*")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Campo para API key
    api_key = st.text_input(
        "OpenAI API Key", 
        type="password",
        help="Ingresa tu clave API de OpenAI"
    )
    
    if api_key:
        openai.api_key = api_key
        client = OpenAI(api_key=api_key)
        st.success("✅ API Key configurada")
    else:
        st.warning("⚠️ Por favor ingresa tu API Key de OpenAI")
    
    st.markdown("---")
    
    # Configuraciones adicionales
    st.subheader("🎛️ Configuraciones de Análisis")
    
    analysis_depth = st.selectbox(
        "Profundidad de análisis",
        ["Básico", "Detallado", "Completo"],
        index=1
    )
    
    max_matches = st.slider(
        "Máximo de coincidencias a mostrar",
        min_value=3,
        max_value=15,
        value=5
    )
    
    confidence_threshold = st.slider(
        "Umbral de confianza (%)",
        min_value=50,
        max_value=95,
        value=70
    )

# Datos de productos (simulando la carga del Excel)
@st.cache_data
def load_product_data():
    """Simula la carga de datos del Excel con algunos productos de ejemplo"""
    # En una implementación real, aquí cargarías el archivo Excel completo
    sample_data = {
        'TIPO_MADERA': [
            'ASERRADA SIN INMUNIZAR', 'ASERRADA INMUNIZADA', 'CILINDRADA INMUNIZADA',
            'ASERRADA SIN INMUNIZAR', 'ASERRADA INMUNIZADA', 'CILINDRADA INMUNIZADA'
        ],
        'PRODUCTO': [
            'TABLAS, TABLILLAS, TABLONES', 'TABLAS, TABLILLAS, TABLONES', 'ALFARDA',
            'TABLAS, TABLILLAS, TABLONES', 'TABLAS, TABLILLAS, TABLONES', 'ESTACON CALIBRADO'
        ],
        'Referencia': [
            'C3PYP1017100', 'C4PYP1017100', 'RA40016300',
            'C3DEK1425100', 'C4DEK1425100', 'RC40009150'
        ],
        'DESCRIPCION': [
            'PISO PARED 10X1.7X100M2 CEP', 'PISO PARED 10X1.7X100M2', 'ALFARDA TRATADA 16X300',
            'TABLA DECK 14X2.5X100', 'TABLA DECK 14X2.5X100', 'CALIBRADO TRATADO 9X150'
        ],
        'ACABADO': [
            'CEPILLADO SIN INMUNIZAR', 'CEPILLADO INMUNIZADA', 'CILINDRADA',
            'CEPILLADO SIN INMUNIZAR', 'CEPILLADO INMUNIZADA', 'CALIBRADA'
        ],
        'USO': [
            'CONSTRUCCION', 'CONSTRUCCION', 'CONSTRUCCION',
            'CONSTRUCCION', 'CONSTRUCCION', 'CONSTRUCCION Y CERCAS'
        ],
        'PRECIO_CALDAS': [23800, 35920, 107920, 5732, 8857, 16699]
    }
    return pd.DataFrame(sample_data)

# Funciones de análisis con OpenAI
def analyze_image_with_openai(image_data: bytes, products_df: pd.DataFrame) -> Dict[str, Any]:
    """Analiza una imagen usando la API de OpenAI Vision"""
    if not api_key:
        return {"error": "API Key no configurada"}
    
    try:
        # Convertir imagen a base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Crear descripción de productos para el contexto
        products_context = create_products_context(products_df)
        
        # Prompt para análisis de imagen
        prompt = f"""
        Analiza esta imagen e identifica cualquier elemento relacionado con productos de madera.
        Busca específicamente:
        - Tipo de madera (aserrada, cilindrada, etc.)
        - Dimensiones visibles
        - Acabados (cepillado, calibrado, rústico, etc.)
        - Usos posibles (construcción, cercas, cultivos, etc.)
        - Características distintivas
        
        Productos disponibles en nuestra base de datos:
        {products_context}
        
        Responde en formato JSON con:
        {{
            "elementos_identificados": ["lista de elementos encontrados"],
            "tipo_madera_probable": "tipo identificado",
            "dimensiones_estimadas": "dimensiones si son visibles",
            "uso_sugerido": "uso probable",
            "productos_relacionados": [
                {{
                    "referencia": "código",
                    "descripcion": "descripción del producto",
                    "confianza": "porcentaje de confianza",
                    "justificacion": "por qué coincide"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=1500
        )
        
        # Intentar parsear la respuesta como JSON
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            return {
                "elementos_identificados": ["Análisis textual disponible"],
                "descripcion_general": response.choices[0].message.content
            }
            
    except Exception as e:
        return {"error": f"Error en análisis de imagen: {str(e)}"}

def analyze_audio_with_openai(audio_data: bytes, products_df: pd.DataFrame) -> Dict[str, Any]:
    """Analiza audio usando la API de OpenAI Whisper"""
    if not api_key:
        return {"error": "API Key no configurada"}
    
    try:
        # Crear archivo temporal en memoria
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.wav"  # OpenAI necesita un nombre con extensión
        
        # Transcribir audio
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        
        # Analizar transcripción
        products_context = create_products_context(products_df)
        
        prompt = f"""
        Analiza esta transcripción de audio y encuentra referencias a productos de madera:
        
        Transcripción: "{transcript.text}"
        
        Productos disponibles:
        {products_context}
        
        Responde en formato JSON con:
        {{
            "transcripcion": "texto transcrito",
            "palabras_clave": ["palabras relacionadas con madera"],
            "productos_mencionados": ["productos específicos mencionados"],
            "productos_relacionados": [
                {{
                    "referencia": "código",
                    "descripcion": "descripción",
                    "confianza": "porcentaje",
                    "justificacion": "por qué coincide"
                }}
            ]
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            return {
                "transcripcion": transcript.text,
                "analisis": response.choices[0].message.content
            }
            
    except Exception as e:
        return {"error": f"Error en análisis de audio: {str(e)}"}

def create_products_context(products_df: pd.DataFrame) -> str:
    """Crea un contexto resumido de productos para la IA"""
    context_parts = []
    for _, row in products_df.head(10).iterrows():  # Limitamos para no exceder tokens
        context_parts.append(
            f"- {row['Referencia']}: {row['DESCRIPCION']} "
            f"({row['TIPO_MADERA']}, {row['ACABADO']}, {row['USO']})"
        )
    return "\n".join(context_parts)

def search_similar_products(query: str, products_df: pd.DataFrame) -> pd.DataFrame:
    """Busca productos similares basado en una consulta"""
    query = query.lower()
    mask = (
        products_df['DESCRIPCION'].str.lower().str.contains(query, na=False) |
        products_df['TIPO_MADERA'].str.lower().str.contains(query, na=False) |
        products_df['PRODUCTO'].str.lower().str.contains(query, na=False) |
        products_df['USO'].str.lower().str.contains(query, na=False)
    )
    return products_df[mask]

# Cargar datos
products_df = load_product_data()

# Interface principal
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📤 Subir Archivos para Análisis")
    
    # Tabs para diferentes tipos de archivos
    tab1, tab2, tab3 = st.tabs(["🖼️ Imágenes", "🎵 Audio", "🔍 Búsqueda Manual"])
    
    with tab1:
        st.subheader("Análisis de Imágenes")
        uploaded_image = st.file_uploader(
            "Sube una imagen de productos de madera",
            type=['png', 'jpg', 'jpeg'],
            help="Formatos soportados: PNG, JPG, JPEG"
        )
        
        if uploaded_image and api_key:
            # Mostrar imagen
            image = Image.open(uploaded_image)
            st.image(image, caption="Imagen cargada", use_column_width=True)
            
            if st.button("🔍 Analizar Imagen", type="primary"):
                with st.spinner("Analizando imagen con IA..."):
                    # Convertir imagen a bytes
                    img_bytes = uploaded_image.getvalue()
                    result = analyze_image_with_openai(img_bytes, products_df)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("✅ Análisis completado!")
                        
                        # Mostrar resultados
                        if "elementos_identificados" in result:
                            st.subheader("🔍 Elementos Identificados")
                            for elemento in result["elementos_identificados"]:
                                st.write(f"• {elemento}")
                        
                        if "productos_relacionados" in result:
                            st.subheader("🎯 Productos Relacionados")
                            for producto in result["productos_relacionados"]:
                                with st.expander(f"📦 {producto.get('referencia', 'N/A')} - Confianza: {producto.get('confianza', 'N/A')}"):
                                    st.write(f"**Descripción:** {producto.get('descripcion', 'N/A')}")
                                    st.write(f"**Justificación:** {producto.get('justificacion', 'N/A')}")
    
    with tab2:
        st.subheader("Análisis de Audio")
        uploaded_audio = st.file_uploader(
            "Sube un archivo de audio",
            type=['wav', 'mp3', 'm4a'],
            help="Formatos soportados: WAV, MP3, M4A"
        )
        
        if uploaded_audio and api_key:
            st.audio(uploaded_audio)
            
            if st.button("🎧 Analizar Audio", type="primary"):
                with st.spinner("Transcribiendo y analizando audio..."):
                    audio_bytes = uploaded_audio.getvalue()
                    result = analyze_audio_with_openai(audio_bytes, products_df)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("✅ Análisis completado!")
                        
                        # Mostrar transcripción
                        if "transcripcion" in result:
                            st.subheader("📝 Transcripción")
                            st.write(result["transcripcion"])
                        
                        # Mostrar palabras clave
                        if "palabras_clave" in result:
                            st.subheader("🔑 Palabras Clave Identificadas")
                            cols = st.columns(3)
                            for i, palabra in enumerate(result["palabras_clave"]):
                                with cols[i % 3]:
                                    st.info(palabra)
                        
                        # Mostrar productos relacionados
                        if "productos_relacionados" in result:
                            st.subheader("🎯 Productos Relacionados")
                            for producto in result["productos_relacionados"]:
                                with st.expander(f"📦 {producto.get('referencia', 'N/A')} - Confianza: {producto.get('confianza', 'N/A')}"):
                                    st.write(f"**Descripción:** {producto.get('descripcion', 'N/A')}")
                                    st.write(f"**Justificación:** {producto.get('justificacion', 'N/A')}")
    
    with tab3:
        st.subheader("Búsqueda Manual de Productos")
        search_query = st.text_input(
            "Buscar productos por descripción, tipo o uso",
            placeholder="Ej: tabla, deck, piso, alfarda, estacón..."
        )
        
        if search_query:
            similar_products = search_similar_products(search_query, products_df)
            
            if not similar_products.empty:
                st.subheader(f"🔍 Resultados de búsqueda ({len(similar_products)} encontrados)")
                st.dataframe(
                    similar_products[['Referencia', 'DESCRIPCION', 'TIPO_MADERA', 'USO', 'PRECIO_CALDAS']],
                    use_container_width=True
                )
            else:
                st.info("No se encontraron productos que coincidan con la búsqueda.")

with col2:
    st.header("📊 Base de Datos de Productos")
    
    # Estadísticas rápidas
    st.metric("Total Productos", len(products_df))
    st.metric("Tipos de Madera", products_df['TIPO_MADERA'].nunique())
    st.metric("Categorías de Producto", products_df['PRODUCTO'].nunique())
    
    # Filtros
    st.subheader("🔧 Filtros")
    
    selected_type = st.selectbox(
        "Tipo de Madera",
        ["Todos"] + list(products_df['TIPO_MADERA'].unique())
    )
    
    selected_use = st.selectbox(
        "Uso",
        ["Todos"] + list(products_df['USO'].unique())
    )
    
    # Aplicar filtros
    filtered_df = products_df.copy()
    if selected_type != "Todos":
        filtered_df = filtered_df[filtered_df['TIPO_MADERA'] == selected_type]
    if selected_use != "Todos":
        filtered_df = filtered_df[filtered_df['USO'] == selected_use]
    
    # Mostrar productos filtrados
    st.subheader("📋 Productos Disponibles")
    if not filtered_df.empty:
        for _, product in filtered_df.head(5).iterrows():
            with st.expander(f"📦 {product['Referencia']}"):
                st.write(f"**Descripción:** {product['DESCRIPCION']}")
                st.write(f"**Tipo:** {product['TIPO_MADERA']}")
                st.write(f"**Acabado:** {product['ACABADO']}")
                st.write(f"**Uso:** {product['USO']}")
                st.write(f"**Precio:** ${product['PRECIO_CALDAS']:,}")
    
    # Información adicional
    st.markdown("---")
    st.info(
        "💡 **Tip:** La IA puede identificar productos de madera en imágenes "
        "analizando dimensiones, acabados, tipos de corte y características visuales. "
        "También puede procesar descripciones de audio para encontrar productos específicos."
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🪵 Identificador de Productos de Madera con IA | "
    f"Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    "</div>",
    unsafe_allow_html=True
)

# Instrucciones de uso
with st.expander("📖 Instrucciones de Uso"):
    st.markdown("""
    ### 🚀 Cómo usar esta aplicación:
    
    1. **Configuración inicial:**
       - Ingresa tu API Key de OpenAI en la barra lateral
       - Ajusta las configuraciones según tus necesidades
    
    2. **Análisis de imágenes:**
       - Sube una imagen que contenga productos de madera
       - La IA identificará tipos, dimensiones y características
       - Recibirás sugerencias de productos relacionados
    
    3. **Análisis de audio:**
       - Graba o sube un archivo de audio describiendo lo que necesitas
       - La IA transcribirá y analizará el contenido
       - Encontrará productos que coincidan con la descripción
    
    4. **Búsqueda manual:**
       - Usa palabras clave para buscar productos específicos
       - Explora la base de datos con filtros
    
    ### 🎯 Casos de uso:
    - Identificar productos de madera en fotos de proyectos
    - Transcribir y procesar pedidos de audio
    - Buscar productos similares o complementarios
    - Obtener información técnica y precios
    """)

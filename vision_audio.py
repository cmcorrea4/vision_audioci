def analyze_image_with_openai(image_data: bytes, products_df: pd.DataFrame) -> Dict[str, Any]:
    """Analiza una imagen con lista de productos usando la API de OpenAI Vision"""
    if not api_key:
        return {"error": "API Key no configurada"}
    
    try:
        # Convertir imagen a base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Prompt específico para extraer listas de productos
        prompt = f"""
        Analiza esta imagen que contiene una lista de productos de madera (factura, cotización, lista de materiales, etc.).
        
        Extrae TODOS los productos de madera mencionados en la imagen con sus:
        - Nombre/descripción del producto
        - Dimensiones (si están visibles)
        - Cantidades
        - Cualquier especificación técnica
        
        Responde en formato JSON con:
        {{
            "productos_encontrados": [
                {{
                    "descripcion": "nombre del producto extraído",
                    "dimensiones": "dimensiones si están visibles",
                    "cantidad": "cantidad si está visible",
                    "especificaciones": "otras especificaciones"
                }}
            ],
            "texto_completo": "todo el texto relevante extraído de la imagen",
            "tipo_documento": "factura/cotización/lista/otro"
        }}
        
        Enfócate en extraer TODOS los productos de madera, tablones, tablas, estacones, alfardas, etc.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
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
            max_tokens=2000
        )
        
        # Intentar parsear la respuesta como JSON
        try:
            result = json.loads(response.choices[0].message.content)
            
            # Agregar matching con catálogo para cada producto encontrado
            if "productos_encontrados" in result:
                for producto in result["productos_encontrados"]:
                    matches = find_similar_products(producto["descripcion"], products_df)
                    producto["productos_similares"] = matches
            
            return result
            
        except json.JSONDecodeError:
            return {
                "productos_encontrados": [],
                "texto_extraido": response.choices[0].message.content,
                "error_parsing": "No se pudo parsear como JSON"
            }
            
    except Exception as e:
        return {"error": f"Error en análisis de imagen: {str(e)}"}

def analyze_audio_with_openai(audio_data: bytes, products_df: pd.DataFrame) -> Dict[str, Any]:
    """Analiza audio con lista de productos usando la API de OpenAI Whisper"""
    if not api_key:
        return {"error": "API Key no configurada"}
    
    try:
        # Crear archivo temporal en memoria
        audio_file = io.BytesIO(audio_data)
        audio_file.name = "audio.wav"
        
        # Transcribir audio
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        
        # Analizar transcripción para extraer productos
        prompt = f"""
        Analiza esta transcripción de audio que contiene una lista de productos de madera.
        
        Transcripción: "{transcript.text}"
        
        Extrae TODOS los productos de madera mencionados con:
        - Nombre/descripción
        - Dimensiones mencionadas
        - Cantidades
        - Especificaciones
        
        Responde en formato JSON con:
        {{
            "transcripcion_completa": "texto completo transcrito",
            "productos_mencionados": [
                {{
                    "descripcion": "producto mencionado",
                    "dimensiones": "dimensiones si se mencionan",
                    "cantidad": "cantidad si se menciona",
                    "especificaciones": "otras especificaciones"
                }}
            ],
            "contexto": "tipo de pedido/consulta/lista"
        }}
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500
        )
        
        try:
            result = json.loads(response.choices[0].message.content)
            
            # Agregar matching con catálogo para cada producto mencionado
            if "productos_mencionados" in result:
                for producto in result["productos_mencionados"]:
                    matches = find_similar_products(producto["descripcion"], products_df)
                    producto["productos_similares"] = matches
            
            return result
            
        except json.JSONDecodeError:
            return {
                "transcripcion_completa": transcript.text,
                "analisis": response.choices[0].message.content
            }
            
    except Exception as e:
        return {"error": f"Error en análisis de audio: {str(e)}"}

def find_similar_products(product_description: str, products_df: pd.DataFrame, max_results: int = 3) -> List[Dict[str, Any]]:
    """Encuentra productos similares en el catálogo basado en descripción"""
    description = product_description.lower()
    
    # Detectar columna de tipo
    tipo_col = 'TIPO MADERA' if 'TIPO MADERA' in products_df.columns else 'TIPO_MADERA'
    
    # Buscar coincidencias por palabras clave
    matches = []
    
    for _, row in products_df.iterrows():
        score = 0
        reasons = []
        
        # Buscar en descripción
        if pd.notna(row['DESCRIPCION']):
            desc_words = row['DESCRIPCION'].lower().split()
            input_words = description.split()
            
            common_words = set(desc_words) & set(input_words)
            if common_words:
                score += len(common_words)
                reasons.append(f"Palabras coincidentes: {', '.join(common_words)}")
        
        # Buscar palabras clave específicas
        keywords = {
            'tabla': ['tabla', 'tablilla', 'tablon'],
            'deck': ['deck'],
            'piso': ['piso', 'pared'],
            'alfarda': ['alfarda'],
            'estacon': ['estacon', 'calibrado'],
            'columna': ['columna'],
            'vareta': ['vareta', 'varillon']
        }
        
        for category, words in keywords.items():
            if any(word in description for word in words):
                if any(word in row['DESCRIPCION'].lower() for word in words):
                    score += 5
                    reasons.append(f"Categoría: {category}")
        
        # Buscar dimensiones (formato NxNxN)
        import re
        dim_pattern = r'(\d+\.?\d*)[xX](\d+\.?\d*)[xX]?(\d+\.?\d*)?'
        input_dims = re.findall(dim_pattern, description)
        desc_dims = re.findall(dim_pattern, row['DESCRIPCION'].lower())
        
        if input_dims and desc_dims:
            if any(dim in desc_dims for dim in input_dims):
                score += 10
                reasons.append("Dimensiones similares")
        
        if score > 0:
            matches.append({
                'referencia': row['Referencia'],
                'descripcion': row['DESCRIPCION'],
                'tipo': row[tipo_col],
                'precio': row['PRECIO_CALDAS'] if pd.notna(row['PRECIO_CALDAS']) else 'No disponible',
                'score': score,
                'razones': '; '.join(reasons)
            })
    
    # Ordenar por score y retornar los mejores
    matches = sorted(matches, key=lambda x: x['score'], reverse=True)
    return matches[:max_results]
    
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
    page_title="Analizador de Listas de Productos de Madera IA",
    page_icon="🪵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🪵 Analizador de Listas de Productos de Madera con IA")
st.markdown("*Extrae listas de productos desde imágenes y audio, encuentra equivalencias en tu catálogo*")

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

# Cargar datos del archivo Excel
@st.cache_data
def load_product_data():
    """Carga los datos del archivo Excel de productos"""
    try:
        # Cargar el archivo Excel desde la raíz
        df = pd.read_excel('GUION PARA IA LISTADO.xlsx')
        
        # Limpiar nombres de columnas
        df.columns = df.columns.str.strip()
        
        # Renombrar columnas para consistencia
        column_mapping = {
            'ACABADO DE LA MADERA': 'ACABADO',
            'ACABADO_DE_LA_MADERA': 'ACABADO'
        }
        df = df.rename(columns=column_mapping)
        
        # Eliminar filas completamente vacías
        df = df.dropna(how='all')
        
        # Filtrar filas que tengan al menos referencia y descripción
        df = df.dropna(subset=['Referencia', 'DESCRIPCION'])
        
        # Limpiar datos de precio
        if 'PRECIO CALDAS' in df.columns:
            df['PRECIO_CALDAS'] = df['PRECIO CALDAS']
        
        df['PRECIO_CALDAS'] = pd.to_numeric(df['PRECIO_CALDAS'], errors='coerce')
        
        st.success(f"✅ Archivo Excel cargado: {len(df)} productos encontrados")
        return df
        
    except FileNotFoundError:
        st.error("❌ No se encontró el archivo 'GUION PARA IA LISTADO.xlsx' en la raíz del proyecto")
        st.info("📁 Asegúrate de que el archivo esté en la misma carpeta que la aplicación")
        
        # Datos de ejemplo como fallback
        sample_data = {
            'TIPO MADERA': ['ASERRADA SIN INMUNIZAR', 'ASERRADA INMUNIZADA'],
            'PRODUCTO': ['TABLAS, TABLILLAS, TABLONES', 'TABLAS, TABLILLAS, TABLONES'],
            'Referencia': ['C3PYP1017100', 'C4PYP1017100'],
            'DESCRIPCION': ['PISO PARED 10X1.7X100M2 CEP', 'PISO PARED 10X1.7X100M2'],
            'ACABADO': ['CEPILLADO SIN INMUNIZAR', 'CEPILLADO INMUNIZADA'],
            'USO': ['CONSTRUCCION', 'CONSTRUCCION'],
            'PRECIO_CALDAS': [23800, 35920]
        }
        return pd.DataFrame(sample_data)
        
    except Exception as e:
        st.error(f"❌ Error cargando archivo Excel: {str(e)}")
        st.info("📋 Usando datos de ejemplo. Revisa el formato del archivo Excel.")
        
        # Datos de ejemplo como fallback
        sample_data = {
            'TIPO MADERA': ['ASERRADA SIN INMUNIZAR', 'ASERRADA INMUNIZADA'],
            'PRODUCTO': ['TABLAS, TABLILLAS, TABLONES', 'TABLAS, TABLILLAS, TABLONES'],
            'Referencia': ['C3PYP1017100', 'C4PYP1017100'],
            'DESCRIPCION': ['PISO PARED 10X1.7X100M2 CEP', 'PISO PARED 10X1.7X100M2'],
            'ACABADO': ['CEPILLADO SIN INMUNIZAR', 'CEPILLADO INMUNIZADA'],
            'USO': ['CONSTRUCCION', 'CONSTRUCCION'],
            'PRECIO_CALDAS': [23800, 35920]
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
    # Usar los nombres de columnas correctos del Excel
    tipo_col = 'TIPO MADERA' if 'TIPO MADERA' in products_df.columns else 'TIPO_MADERA'
    
    for _, row in products_df.head(10).iterrows():  # Limitamos para no exceder tokens
        context_parts.append(
            f"- {row['Referencia']}: {row['DESCRIPCION']} "
            f"({row[tipo_col]}, {row['ACABADO']}, {row['USO']})"
        )
    return "\n".join(context_parts)

def search_similar_products(query: str, products_df: pd.DataFrame) -> pd.DataFrame:
    """Busca productos similares basado en una consulta"""
    query = query.lower()
    
    # Detectar nombres de columnas dinámicamente
    tipo_col = 'TIPO MADERA' if 'TIPO MADERA' in products_df.columns else 'TIPO_MADERA'
    
    mask = (
        products_df['DESCRIPCION'].str.lower().str.contains(query, na=False) |
        products_df[tipo_col].str.lower().str.contains(query, na=False) |
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
        st.subheader("Análisis de Listas de Productos en Imágenes")
        uploaded_image = st.file_uploader(
            "Sube una imagen con lista de productos (factura, cotización, lista de materiales, etc.)",
            type=['png', 'jpg', 'jpeg'],
            help="Formatos soportados: PNG, JPG, JPEG. La IA extraerá todos los productos de madera de la lista."
        )
        
        if uploaded_image and api_key:
            # Mostrar imagen
            image = Image.open(uploaded_image)
            st.image(image, caption="Imagen cargada", use_container_width=True)
            
            if st.button("🔍 Analizar Lista de Productos", type="primary"):
                with st.spinner("Extrayendo productos de la imagen..."):
                    # Convertir imagen a bytes
                    img_bytes = uploaded_image.getvalue()
                    result = analyze_image_with_openai(img_bytes, products_df)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("✅ Análisis completado!")
                        
                        # Mostrar tipo de documento
                        if "tipo_documento" in result:
                            st.info(f"📄 Documento detectado: {result['tipo_documento']}")
                        
                        # Mostrar productos encontrados
                        if "productos_encontrados" in result and result["productos_encontrados"]:
                            st.subheader("📋 Productos Extraídos de la Imagen")
                            
                            for i, producto in enumerate(result["productos_encontrados"], 1):
                                with st.expander(f"📦 Producto {i}: {producto.get('descripcion', 'Sin descripción')[:50]}..."):
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.write("**Información Extraída:**")
                                        st.write(f"• **Descripción:** {producto.get('descripcion', 'N/A')}")
                                        st.write(f"• **Dimensiones:** {producto.get('dimensiones', 'N/A')}")
                                        st.write(f"• **Cantidad:** {producto.get('cantidad', 'N/A')}")
                                        st.write(f"• **Especificaciones:** {producto.get('especificaciones', 'N/A')}")
                                    
                                    with col2:
                                        st.write("**Productos Similares en Catálogo:**")
                                        if "productos_similares" in producto and producto["productos_similares"]:
                                            for match in producto["productos_similares"]:
                                                st.info(f"**{match['referencia']}** - {match['descripcion'][:40]}...\n"
                                                        f"💰 Precio: ${match['precio']}\n"
                                                        f"🎯 Coincidencia: {match['razones']}")
                                        else:
                                            st.warning("No se encontraron productos similares")
                        
                        # Mostrar texto completo extraído
                        if "texto_completo" in result:
                            with st.expander("📝 Texto Completo Extraído"):
                                st.text_area("Contenido:", result["texto_completo"], height=200)
                        
                        # Si hay error de parsing, mostrar contenido raw
                        if "texto_extraido" in result:
                            with st.expander("📝 Análisis Completo"):
                                st.write(result["texto_extraido"])
    
    with tab2:
        st.subheader("Análisis de Listas de Productos en Audio")
        uploaded_audio = st.file_uploader(
            "Sube un archivo de audio con lista de productos o pedido",
            type=['wav', 'mp3', 'm4a'],
            help="Formatos soportados: WAV, MP3, M4A. La IA transcribirá y extraerá todos los productos mencionados."
        )
        
        if uploaded_audio and api_key:
            st.audio(uploaded_audio)
            
            if st.button("🎧 Analizar Lista de Audio", type="primary"):
                with st.spinner("Transcribiendo y extrayendo productos..."):
                    audio_bytes = uploaded_audio.getvalue()
                    result = analyze_audio_with_openai(audio_bytes, products_df)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("✅ Análisis completado!")
                        
                        # Mostrar transcripción
                        if "transcripcion_completa" in result:
                            st.subheader("📝 Transcripción Completa")
                            st.text_area("Audio transcrito:", result["transcripcion_completa"], height=100)
                        
                        # Mostrar contexto
                        if "contexto" in result:
                            st.info(f"📄 Contexto detectado: {result['contexto']}")
                        
                        # Mostrar productos mencionados
                        if "productos_mencionados" in result and result["productos_mencionados"]:
                            st.subheader("📋 Productos Mencionados en Audio")
                            
                            for i, producto in enumerate(result["productos_mencionados"], 1):
                                with st.expander(f"🎤 Producto {i}: {producto.get('descripcion', 'Sin descripción')[:50]}..."):
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.write("**Información Mencionada:**")
                                        st.write(f"• **Descripción:** {producto.get('descripcion', 'N/A')}")
                                        st.write(f"• **Dimensiones:** {producto.get('dimensiones', 'N/A')}")
                                        st.write(f"• **Cantidad:** {producto.get('cantidad', 'N/A')}")
                                        st.write(f"• **Especificaciones:** {producto.get('especificaciones', 'N/A')}")
                                    
                                    with col2:
                                        st.write("**Productos Similares en Catálogo:**")
                                        if "productos_similares" in producto and producto["productos_similares"]:
                                            for match in producto["productos_similares"]:
                                                st.info(f"**{match['referencia']}** - {match['descripcion'][:40]}...\n"
                                                        f"💰 Precio: ${match['precio']}\n"
                                                        f"🎯 Coincidencia: {match['razones']}")
                                        else:
                                            st.warning("No se encontraron productos similares")
                        
                        # Si hay análisis sin parsing
                        if "analisis" in result:
                            with st.expander("📝 Análisis Completo"):
                                st.write(result["analisis"])
    
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
                
                # Detectar nombres de columnas para mostrar
                tipo_col = 'TIPO MADERA' if 'TIPO MADERA' in similar_products.columns else 'TIPO_MADERA'
                columns_to_show = ['Referencia', 'DESCRIPCION', tipo_col, 'USO', 'PRECIO_CALDAS']
                
                # Filtrar solo las columnas que existen
                available_columns = [col for col in columns_to_show if col in similar_products.columns]
                
                st.dataframe(
                    similar_products[available_columns],
                    use_container_width=True
                )
            else:
                st.info("No se encontraron productos que coincidan con la búsqueda.")

with col2:
    st.header("📊 Base de Datos de Productos")
    
    # Detectar nombres de columnas dinámicamente
    tipo_col = 'TIPO MADERA' if 'TIPO MADERA' in products_df.columns else 'TIPO_MADERA'
    
    # Estadísticas rápidas
    st.metric("Total Productos", len(products_df))
    st.metric("Tipos de Madera", products_df[tipo_col].nunique())
    st.metric("Categorías de Producto", products_df['PRODUCTO'].nunique())
    
    # Filtros
    st.subheader("🔧 Filtros")
    
    selected_type = st.selectbox(
        "Tipo de Madera",
        ["Todos"] + list(products_df[tipo_col].unique())
    )
    
    selected_use = st.selectbox(
        "Uso",
        ["Todos"] + list(products_df['USO'].unique())
    )
    
    # Aplicar filtros
    filtered_df = products_df.copy()
    if selected_type != "Todos":
        filtered_df = filtered_df[filtered_df[tipo_col] == selected_type]
    if selected_use != "Todos":
        filtered_df = filtered_df[filtered_df['USO'] == selected_use]
    
    # Mostrar productos filtrados
    st.subheader("📋 Productos Disponibles")
    if not filtered_df.empty:
        for _, product in filtered_df.head(5).iterrows():
            with st.expander(f"📦 {product['Referencia']}"):
                st.write(f"**Descripción:** {product['DESCRIPCION']}")
                st.write(f"**Tipo:** {product[tipo_col]}")
                st.write(f"**Acabado:** {product['ACABADO']}")
                st.write(f"**Uso:** {product['USO']}")
                if pd.notna(product['PRECIO_CALDAS']):
                    st.write(f"**Precio:** ${product['PRECIO_CALDAS']:,.0f}")
    
    # Información adicional
    st.markdown("---")
    st.info(
        "💡 **Tip:** La IA puede extraer listas completas de productos de madera desde imágenes "
        "(facturas, cotizaciones, listas de materiales) y desde audio (pedidos, consultas). "
        "Luego busca productos similares en el catálogo y los asocia automáticamente."
    )

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "🪵 Analizador de Listas de Productos de Madera con IA | "
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
    
    2. **Análisis de listas en imágenes:**
       - Sube una imagen con una lista de productos (factura, cotización, lista de materiales)
       - La IA extraerá TODOS los productos de madera de la lista
       - Recibirás productos similares de tu catálogo para cada item encontrado
    
    3. **Análisis de listas en audio:**
       - Graba o sube un archivo de audio mencionando productos que necesitas
       - La IA transcribirá y extraerá todos los productos mencionados
       - Encontrará productos similares en tu catálogo
    
    4. **Búsqueda manual:**
       - Usa palabras clave para buscar productos específicos
       - Explora la base de datos con filtros
    
    ### 🎯 Casos de uso principales:
    - Procesar facturas o cotizaciones de competencia
    - Extraer listas de materiales de proyectos
    - Transcribir y procesar pedidos de audio
    - Encontrar equivalencias en tu catálogo
    - Generar cotizaciones basadas en listas existentes
    """)

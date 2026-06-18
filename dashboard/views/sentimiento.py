import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.sentiment_pipeline import load_rslora_model, analyze_sentiment

st.title("📰 Fase 1: Análisis de Sentimiento")
st.markdown("Monitorización y análisis del humor de mercado mediante PLN usando **FinBERT adaptado con rsLoRA**.")

st.divider()

with st.expander("ℹ️ ¿Cómo funciona este módulo?", expanded=False):
    st.info("""
    * 🎯 **Objetivo:** Este módulo actúa como el sensor cualitativo del ecosistema D.R.E.A.M..
    * ⚙️ **Tecnología:** Procesa texto no estructurado, como noticias financieras y titulares, utilizando el modelo de lenguaje FinBERT. Para ser computacionalmente eficiente, el modelo ha sido ajustado mediante la técnica de adaptación rsLoRA, entrenando solo una fracción mínima de los parámetros y preservando el conocimiento financiero original.
    * 📈 **Salida:** Convierte la ambigüedad del lenguaje en una señal matemática, generando un indicador de polaridad continuo donde -1 indica un sentimiento fuertemente negativo, 0 neutralidad y 1 un sentimiento fuertemente positivo.
    """)

@st.cache_resource(show_spinner="Cargando modelo base FinBERT y pesos del adaptador rsLoRA en memoria...")
def get_nlp_model():
    base_name = "ProsusAI/finbert"
    lora_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "finbert_rslora_finetuned")
    return load_rslora_model(base_name, lora_path)

# Precargar el modelo para que esté listo en caché
model, tokenizer, device = get_nlp_model()

with st.container():
    st.subheader("Análisis de Titulares en Tiempo Real")
    
    # El placeholder ahora está en inglés ya que FinBERT procesa en inglés
    headline = st.text_input(
        "📝 Introduce un titular financiero en inglés para evaluar con FinBERT + rsLoRA:", 
        "Nvidia shares surge 15% as data center revenue triples estimates"
    )

    if st.button("Analizar Sentimiento 🧠", type="primary"):
        if model is None:
            st.error("Error: No se pudo cargar el modelo rsLoRA. Verifica las rutas en la carpeta models/.")
        else:
            with st.spinner("Ejecutando inferencia real en FinBERT (rsLoRA)..."):
                # 1. INFERENCIA BASE
                df_input = pd.DataFrame([{"text": headline}])
                df_out = analyze_sentiment(df_input, model, tokenizer, device, temperature=1.0)
                
                score = float(df_out.iloc[0]["sentiment_score"])
                p_neg = float(df_out.iloc[0]["prob_negative"])
                p_neu = float(df_out.iloc[0]["prob_neutral"])
                p_pos = float(df_out.iloc[0]["prob_positive"])
                
                if score > 0.15:
                    sentiment_label = "POSITIVO"
                elif score < -0.15:
                    sentiment_label = "NEGATIVO"
                else:
                    sentiment_label = "NEUTRO"
            
            with st.spinner("Calculando impacto por token (Técnica de Oclusión XAI)..."):
                # 2. INTERPRETABILIDAD (OCLUSIÓN POR PALABRA)
                raw_words = headline.split()
                STOPWORDS = {"a", "an", "the", "and", "or", "but", "if", "because", "as", "what", "which", "this", "that", "these", "those", "then", "just", "so", "than", "such", "both", "through", "about", "for", "is", "are", "was", "were", "be", "been", "being", "in", "on", "at", "to", "from", "by", "with", "of", "it", "its"}
                
                words = []
                importance = []
                
                # Limitamos a 60 palabras para no saturar la inferencia en vivo
                if len(raw_words) > 0 and len(raw_words) <= 60:
                    occluded_texts = []
                    for i, w in enumerate(raw_words):
                        w_clean = "".join(c for c in w.lower() if c.isalnum())
                        if w_clean not in STOPWORDS and len(w_clean) > 0:
                            words.append(w)
                            occ_words = raw_words[:i] + raw_words[i+1:]
                            occluded_texts.append(" ".join(occ_words))
                    
                    if occluded_texts:
                        df_occ_in = pd.DataFrame({"text": occluded_texts})
                        df_occ_out = analyze_sentiment(df_occ_in, model, tokenizer, device, temperature=1.0, batch_size=32)
                        
                        for i, occ_score in enumerate(df_occ_out["sentiment_score"]):
                            # Contribución = Baseline Score - Occluded Score
                            importance.append(score - float(occ_score))

            st.success("✅ Análisis completado con éxito.")
            
            # DISEÑO ELEGANTE CON PESTAÑAS PARA LOS RESULTADOS
            tab_resumen, tab_xai, tab_fuerzas = st.tabs(["📊 Resumen de Inferencia", "🔍 Atribución (Barras)", "🧭 Mapa de Fuerzas (Treemap)"])
            
            with tab_resumen:
                col_metric, col_gauge = st.columns([1, 2], gap="large")
                
                with col_metric:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.metric(label="Polaridad Detectada (Score)", 
                              value=f"{score:.4f}", 
                              delta=sentiment_label, 
                              delta_color="normal" if score > 0 else "inverse" if score < 0 else "off")
                    
                    st.markdown(f"**Clasificación Final:** `{sentiment_label}`")
                    st.caption("Confianza del modelo:")
                    st.progress(max(p_pos, p_neg, p_neu), text=f"{max(p_pos, p_neg, p_neu):.1%} de confianza en la clase más probable")
                    
                with col_gauge:
                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = score,
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': "Termómetro de Mercado", 'font': {'size': 18}},
                        gauge = {
                            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "rgba(0,0,0,0)"},
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [-1, -0.2], 'color': "rgba(255, 99, 132, 0.4)"},
                                {'range': [-0.2, 0.2], 'color': "rgba(201, 203, 207, 0.4)"},
                                {'range': [0.2, 1], 'color': "rgba(75, 192, 192, 0.4)"}
                            ],
                            'threshold': {
                                'line': {'color': "gray", 'width': 5},
                                'thickness': 0.75,
                                'value': score
                            }
                        }
                    ))
                    # Aumentamos el margen superior (t=60) para que no se corte el título
                    fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=60, b=20))
                    st.plotly_chart(fig_gauge, theme="streamlit", use_container_width=True)

                st.divider()
                
                # Gráfico de probabilidades horizontales refinado
                st.subheader("Distribución de Probabilidades Softmax")
                probs_df = pd.DataFrame({
                    "Clase": ["Negativo", "Neutro", "Positivo"],
                    "Probabilidad": [p_neg, p_neu, p_pos]
                })
                
                fig_probs = px.bar(
                    probs_df, 
                    x="Probabilidad", 
                    y="Clase", 
                    orientation='h', 
                    color="Clase",
                    color_discrete_map={
                        "Negativo": "#FF4B4B",
                        "Neutro": "gray",
                        "Positivo": "#1B9C85"
                    },
                    text="Probabilidad"
                )
                fig_probs.update_traces(texttemplate='%{text:.2%}', textposition='outside', width=0.6)
                fig_probs.update_layout(height=250, showlegend=False, xaxis=dict(range=[0, 1.1]), margin=dict(t=10, b=10))
                st.plotly_chart(fig_probs, theme="streamlit", use_container_width=True)

            with tab_xai:
                st.subheader("Atribución de Sentimiento por Token")
                st.markdown("Este gráfico muestra cómo cada palabra específica influye en la decisión final del modelo. Las barras hacia la **derecha (verdes)** empujan el sentimiento a territorio Positivo, mientras que las barras hacia la **izquierda (rojas)** lo hunden a Negativo.")
                
                if len(words) > 0 and len(words) <= 60:
                    # Invertimos el orden para que la frase se lea de arriba a abajo en el eje Y
                    df_tokens = pd.DataFrame({
                        "Palabra": [f"{i}: {w}" for i, w in enumerate(words)],
                        "Impacto": importance,
                        "Orden": list(range(len(words)))
                    }).sort_values(by="Orden", ascending=False)
                    
                    # Colores condicionales
                    df_tokens["Color"] = df_tokens["Impacto"].apply(lambda x: "#1B9C85" if x >= 0 else "#FF4B4B")
                    
                    fig_tokens = go.Figure()
                    fig_tokens.add_trace(go.Bar(
                        x=df_tokens["Impacto"],
                        y=df_tokens["Palabra"],
                        orientation='h',
                        marker_color=df_tokens["Color"],
                        text=[f"{val:+.3f}" for val in df_tokens["Impacto"]],
                        textposition="outside"
                    ))
                    
                    fig_tokens.update_layout(
                        height=max(400, len(words)*30),
                        xaxis_title="Contribución Marginal al Score (-1 a 1)",
                        yaxis_title="Tokens",
                        showlegend=False,
                        margin=dict(l=10, r=10, t=30, b=10)
                    )
                    
                    st.plotly_chart(fig_tokens, theme="streamlit", use_container_width=True)
                else:
                    st.warning("El texto es demasiado largo para visualizar la atribución por token en tiempo real (máx 60 palabras) o está vacío.")
                    
            with tab_fuerzas:
                st.subheader("Mapa de Fuerzas Semánticas")
                st.markdown("Visualiza el peso **absoluto** de cada palabra categorizada en Impulsos Positivos vs Presiones Negativas.")
                
                if len(words) > 0 and len(words) <= 60:
                    df_tree = pd.DataFrame({
                        "Palabra": words,
                        "Impacto_Raw": importance
                    })
                    
                    # Categorizar en Positivo o Negativo
                    df_tree["Dirección"] = df_tree["Impacto_Raw"].apply(lambda x: "Impulso Positivo" if x >= 0 else "Presión Negativa")
                    
                    # Para el Treemap, necesitamos magnitud (valor absoluto) para determinar el tamaño
                    df_tree["Magnitud"] = df_tree["Impacto_Raw"].abs()
                    
                    # Descartar palabras sin ningún impacto para limpiar el gráfico
                    df_tree = df_tree[df_tree["Magnitud"] > 0.001]
                    
                    if not df_tree.empty:
                        fig_tree = px.treemap(
                            df_tree, 
                            path=['Dirección', 'Palabra'], 
                            values='Magnitud',
                            color='Dirección',
                            color_discrete_map={
                                "Impulso Positivo": "#1B9C85",
                                "Presión Negativa": "#FF4B4B",
                                "(?)": "gray"
                            },
                            title="Jerarquía de Impacto Semántico"
                        )
                        fig_tree.update_layout(height=450, margin=dict(t=40, l=10, r=10, b=10))
                        # Mejorar legibilidad del texto
                        fig_tree.data[0].textinfo = 'label+value+percent parent'
                        st.plotly_chart(fig_tree, theme="streamlit", use_container_width=True)
                    else:
                        st.info("No se detectaron palabras con un impacto lo suficientemente grande como para desglosar.")
                else:
                    st.warning("El texto es demasiado largo (máx 60 palabras).")

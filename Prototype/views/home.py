import streamlit as st
import pandas as pd
import time
from services.pipeline import FraudDetectionPipeline
from services.history_service import HistoryService

@st.cache_resource(show_spinner=False)
def get_pipeline():
    return FraudDetectionPipeline()

def render_home():
    pipeline = get_pipeline()
    history_service = HistoryService()
    
    st.markdown("## Fraudulent Review Classification")
    st.markdown("Enter a review or upload a file to detect deceptive patterns.")
    
    with st.expander("⚙️ Model Configuration (Select Active Models)", expanded=True):
        col_m1, col_m2, col_m3 = st.columns(3)
        use_hybrid = col_m1.checkbox("Hybrid Model (ABSA + Emotion)", value=True)
        use_absa = col_m2.checkbox("ABSA-only model", value=False)
        use_emo = col_m3.checkbox("Emotion-only model", value=False)
    
    models_selected = use_hybrid or use_absa or use_emo

    tab_text, tab_file = st.tabs(["📝 Single Text Input", "📂 File Upload"])

    # SINGLE TEXT
    with tab_text:
        review_text = st.text_area("Review Text", height=150, placeholder="Paste review here...")
        
        if st.button("Analyse Text", type="primary"):
            if not models_selected:
                st.warning("Please select at least one model in the configuration above to proceed.")
            elif not review_text:
                st.error("Please enter text.")
            else:
                with st.spinner("Processing ..."):
                    results = pipeline.predict_single(
                        review_text, 
                        use_hybrid=use_hybrid, 
                        use_absa=use_absa, 
                        use_emo=use_emo
                    )
                    history_service.add_record(review_text, results)
                
                display_results(results)

        st.divider()
        
        col_h1, col_h2 = st.columns([8, 1])
        with col_h1:
            st.subheader("📜 Recent History")
        with col_h2:
            if st.button("Clear History"):
                history_service.clear_history()
                st.rerun()

        history = history_service.get_history()
        
        if not history:
            st.caption("No history yet. Your past analyses will appear here.")
        else:
            for i, record in enumerate(history):
                label = f"{record['timestamp']} - {record['text'][:60]}..."
                with st.expander(label, expanded=False):
                    st.text(f"Full Input: {record['text']}")
                    st.caption("Results snapshot:")
                    display_results(record['results'])

    # FILE UPLOAD
    with tab_file:
        uploaded_file = st.file_uploader("Upload CSV / Excel", type=['csv', 'xlsx'])
        
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.write("Preview:", df.head(3))
            
            default_col_index = 0
            for i, col in enumerate(df.columns):
                if str(col).lower() in ['review', 'text', 'content', 'body']:
                    default_col_index = i
            
            text_col = st.selectbox("Select Review Column:", df.columns, index=default_col_index)
            
            if st.button("Process", type="primary"):
                if not models_selected:
                    st.warning("⚠️ Please select at least one model in the configuration above to proceed.")
                else:
                    lock_ui = st.empty()
                    lock_ui.markdown("""
                        <style>
                            div[data-testid="stSidebarNav"], 
                            div[role="radiogroup"] {
                                pointer-events: none;
                                opacity: 0.4;
                                filter: grayscale(100%);
                                cursor: not-allowed;
                            }
                        </style>
                        """, unsafe_allow_html=True)

                    try:
                        with st.spinner(f"Processing {len(df)} reviews using selected models..."):
                            result_df = pipeline.predict_batch(
                                df, 
                                text_col, 
                                use_hybrid=use_hybrid, 
                                use_absa=use_absa, 
                                use_emo=use_emo
                            )
                            history_service.add_batch_record(uploaded_file.name, result_df, text_col)
                        st.success("Analysis Complete!")
                        render_batch_table(result_df, text_col, key_prefix="current_run")
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
                    finally:
                        lock_ui.empty()


        st.divider()
        col_b1, col_b2 = st.columns([6, 1])
        with col_b1: 
            st.markdown("""
                <h3 style='margin: 0; font-size: 22px;'>
                    📂 Batch Processing History 
                    <span style='font-size: 16px; font-weight: normal;'>
                        (Last 5)
                    </span>
                </h3>
            """, unsafe_allow_html=True)
        with col_b2:
            if st.button("Clear Batch History"):
                history_service.clear_batch_history()
                st.rerun()

        batch_hist = history_service.get_batch_history()
        if not batch_hist:
            st.caption("No files processed yet.")
        else:
            for i, batch in enumerate(batch_hist): 
                label = f"{batch['timestamp']} | 📄 {batch['filename']} ({batch['count']} rows)"
                
                with st.expander(label):
                    st.write(f"**Processed Data:**")
                    saved_df = batch['df']
                    saved_col_name = batch['text_col']
                    render_batch_table(saved_df, saved_col_name, key_prefix=f"hist_{i}")

def display_results(results):
    st.markdown("### 📊 Analysis Results")
    
    if not results:
        st.error("No results returned. Check model loading.")
        return

    cols = st.columns(len(results))
    
    for i, (model_name, output) in enumerate(results.items()):
        with cols[i]:

            if isinstance(output, str) or "error" in output:
                error_msg = output if isinstance(output, str) else output["error"]
                st.error(f"{model_name}: {error_msg}")
                continue
                
            verdict = output["verdict"]
            confidence = output["confidence"]

            score = int(confidence * 100)

            if verdict == "FRAUD":
                color = "#d50000"
            else:
                color = "#00c853"
            

            st.markdown(f"""
            <div style="background-color: #262730; padding: 15px; border-radius: 10px; border-top: 5px solid {color}; text-align: center;">
                <h4 style="margin:0; color: #FAFAFA; font-size: 16px;">{model_name}</h4>
                <h2 style="color: {color}; margin-top: 10px;">{verdict}</h2>
                <h3 style="color: {color}; margin: 5px 0; font-size: 42px;">{score}%</h3>
                <p style="margin: 0; font-size: 12px; color: #9ca3af;">Confidence Score</p>
            </div>
            """, unsafe_allow_html=True)
            
            '''
            st.markdown(f"""
            <div style="background-color: #262730; padding: 15px; border-radius: 10px; border-top: 5px solid {color}; text-align: center;">
                <h4 style="margin:0; color: #FAFAFA; font-size: 16px;">{model_name}</h4>
                <h1 style="color: {color}; margin: 5px 0; font-size: 42px;">{score}%</h1>
                <p style="margin: 0; font-size: 12px; color: #9ca3af;">Confidence Score</p>
                <h4 style="color: {color}; margin-top: 10px;">{verdict}</h4>
            </div>
            """, unsafe_allow_html=True)
            '''


def render_batch_table(df, text_col, key_prefix="default"):
    cols_to_show = []
    preferred_order = [
        text_col, 
        'Hybrid_Confidence', 'Hybrid_Verdict', 
        'ABSA_Confidence', 'ABSA_Verdict', 
        'Emotion_Confidence', 'Emotion_Verdict'
    ]
    for col in preferred_order:
        if col in df.columns: cols_to_show.append(col)
        
    st.dataframe(df[cols_to_show], width="stretch")
    
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV", 
        data=csv, 
        file_name="results.csv", 
        mime="text/csv", 
        key=f"dl_{key_prefix}_{id(df)}"
    )
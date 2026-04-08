import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def render_experiments():
    st.markdown("## Model Performance & Evaluation")
    st.markdown("""
    This section summarizes the experimental results of the proposed **Hybrid Architecture (ABSA + Emotion)** compared to single-domain baselines.
    """)

    st.divider()

    st.subheader("Model Test Results")
    
    results_data = {
        "Model": ["Hybrid", "ABSA Baseline", "Emotion Baseline"],
        "Accuracy":  [0.8625, 0.8375, 0.7542],
        "F1-Score":  [0.8631, 0.8408, 0.7468],
        "Precision": [0.8595, 0.8240, 0.7699],
        "Recall":    [0.8667, 0.8583, 0.7250]
    }
    
    df_results = pd.DataFrame(results_data)

    df_melted = df_results.melt(id_vars="Model", var_name="Metric", value_name="Score")

    col_chart, col_df = st.columns([1.5, 1])

    with col_chart:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        sns.set_theme(style="whitegrid")
        
        sns.barplot(
            data=df_melted, 
            x="Metric", 
            y="Score", 
            hue="Model", 
            palette="viridis", 
            ax=ax
        )
        
        ax.set_ylim(0.5, 1.0)
        ax.set_title("Performance Comparison by Metric", fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)

        st.pyplot(fig)

    with col_df:
        st.write("### Detailed Scores")
        st.dataframe(
            df_results.style.background_gradient(cmap="Greens", subset=["Accuracy", "F1-Score", "Precision", "Recall"]),
            width="stretch"
        )

    st.divider()

    st.subheader("Confusion Matrix Analysis")
    st.markdown("Visualizing True Positives vs False Positives for the Hybrid Model.")

    col_cm1, col_cm2 = st.columns([1, 1])

    with col_cm1:
        cm_matrix = np.array([[103, 17], [16, 104]])
        labels = ["Genuine", "Fraud"]

        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        
        sns.heatmap(
            cm_matrix, 
            annot=True, 
            fmt='d',
            cmap="Blues", 
            xticklabels=labels, 
            yticklabels=labels,
            cbar=False,
            ax=ax_cm,
            annot_kws={"size": 14, "weight": "bold"}
        )
        
        ax_cm.set_xlabel("Predicted Label", fontsize=12)
        ax_cm.set_ylabel("True Label", fontsize=12)
        ax_cm.set_title("Hybrid Model Confusion Matrix", fontsize=14, fontweight='bold')
        
        st.pyplot(fig_cm)

    with col_cm2:
        st.write("#### Interpretation")
        st.write(f"- **True Positives (Fraud detected as Fraud):** {cm_matrix[1][1]}")
        st.write(f"- **False Negatives (Fraud missed):** {cm_matrix[1][0]}")
        st.write(f"- **False Positives (Genuine flagged as Fraud):** {cm_matrix[0][1]}")
        st.write(f"- **True Negatives (Genuine detected as Genuine):** {cm_matrix[0][0]}")
        

    st.divider()
    
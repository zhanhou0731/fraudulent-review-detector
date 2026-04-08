import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import settings

def render_docs():  
    st.markdown("""
    Integrate the **Fraudulent Review Detection** model directly into your own applications. 
    This API accepts a JSON payload containing the review text and returns fraud verdict and probability.
    """)

    api_url = settings.PUBLIC_API_REQUEST_URL
    docs_url = settings.PUBLIC_API_DOCS_URL

    st.success(f"**🚀 Live Endpoint:** `{api_url}`")

    tab_single, tab_batch = st.tabs(["⚡ Single Prediction", "📦 Batch Processing"])

    with tab_single:
        st.subheader("Endpoint: `/predict`")
        st.caption("Analyze a single review in real-time.")
        st.subheader("📝 Request Format (JSON)")
        st.markdown("To use the API, send a `POST` request with the following JSON structure:")
    
        st.code("""
    {
      "text": "We completed our second stay at the Fairmont Chicago. The check-in process was quick and painless but the front desk staff was almost disinterested in the whole procedure. That is fine when I am traveling on business, but on leisure trips I would appreciate a warmer greeting. The hotel was hosting a convention at the same time as our stay, but other than seeing some signage/attendees in the lobby, we wouldn't have noticed. This is probably due to the separation of the conference areas from the guest rooms and the 8 speedy elevators.",
      "model_type": "hybrid",
      "metadata": {
        "review_id": "R8821",
        "source": "mobile_app"
      }
    }
        """, language="json")
    
        with st.expander("📌 View Parameter Details", expanded=True):
            st.markdown("""
            | Parameter | Type | Required? | Description |
            | :--- | :--- | :--- | :--- |
            | **`text`** | `string` | ✅ Yes | The full content of the review you want to analyze. |
            | **`model_type`** | `string` | ❌ No | Selects the model to use. Defaults to `"hybrid"`. **Options:** `"hybrid"`, `"absa"`, `"emotion"` |
            | **`metadata`** | `dict` | ❌ No | Any optional data you want to pass through (e.g., User ID, Timestamp). This is returned back in the response. |
            """)
    
    
    
        st.subheader("✅ Response Format (JSON)")
        st.markdown("The API returns a JSON object with the verdict and confidence score.")
    
        st.code("""
    {
      "status": "success",
      "model_used": "Hybrid Model",
      "result": {
        "verdict": "GENUINE",
        "confidence_score": 0.9894
      },
      "metadata": {
        "review_id": "R8821",
        "source": "mobile_app"
      }
    }
        """, language="json")
    
    with tab_batch:
        st.subheader("Endpoint: `/predict-batch`")
        st.caption("Process multiple reviews in one request.")

        st.subheader("📝 Request Format (JSON)")
        st.markdown("To use the API, send a `POST` request with the following JSON structure:")
    
        st.code("""
{
  "model_type": "hybrid",
  "reviews": [
    { "id": "r0001", "text": "Great stay!" },
    { "id": "r0002", "text": "Terrible food." }
  ]
}
        """, language="json")
    
        with st.expander("📌 View Parameter Details", expanded=True):
            st.markdown("""
            | Parameter | Type | Required? | Description |
            | :--- | :--- | :--- | :--- |
            | **`reviews`** | `list` | ✅ Yes | A list of objects containing `text` and optional `id`. |
            | **`model_type`** | `str` | ❌ No | The model to use for the entire batch. Defaults to `"hybrid"`. **Options:** `"hybrid"`, `"absa"`, `"emotion"` |
            """)
    
    
    
        st.subheader("✅ Response Format (JSON)")
        st.markdown("The API returns a JSON object with the verdict and confidence score.")
    
        st.code("""
    {
  "status": "success",
  "total_processed": 2,
  "results": [
    {
      "id": "row_1",
      "verdict": "GENUINE",
      "confidence_score": 0.95
    },
    {
      "id": "row_2",
      "verdict": "FRAUD",
      "confidence_score": 0.88
    }
  ]
}
        """, language="json")
        
        
    st.divider()


    st.subheader("How to use the API")
    tab1, tab2, tab3 = st.tabs(["☁️ Python (Cloud)", "🐳 Docker (Local)", "📄 Interactive Swagger"])

    with tab1:
        st.markdown("#### Python Request Example")
        code_python = f"""import requests

url = "{api_url}/predict"
#use [YOUR_IP_ADDRESS]:8000/predict if you deploy the API on your machine

payload = {{
    "text": "The hotel was okay, but the staff was rude.",
    "model_type": "hybrid",
    "metadata": {{ "source": "streamlit_app" }}
}}

try:
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        print("Success:", response.json())
    else:
        print("Error:", response.text)
except Exception as e:
    print("Connection failed:", e)"""
        st.code(code_python, language="python")

    with tab2:
        st.markdown("#### 1. Pull the Image")
        st.code("docker pull zhanhou0731/fraudulent-review-detector:v1.3", language="bash")
        
        st.markdown("#### 2. Run the Container")
        st.code("docker run -p 8000:8000 zhanhou0731/fraudulent-review-detector:v1.3", language="bash")
        
        st.markdown("#### 3. Access Local API")
        st.code("http://[YOUR IP ADDRESS]:8000/predict", language="text")

    with tab3:
        st.info("The Swagger UI allows you to test the API directly from your browser without writing code.")
        
        st.link_button("Open Interactive Swagger UI", docs_url, type="primary")
        
        st.image("https://fastapi.tiangolo.com/img/index/index-03-swagger-02.png", caption="Swagger UI Preview")
    
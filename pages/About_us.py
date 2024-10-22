import streamlit as st

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="My Streamlit App"
)
# endregion <--------- Streamlit App Configuration --------->

st.title("About this App")

st.write("""
    ## Project Scope:
    This project aims to leverage large language models (LLMs) and FAISS (Facebook AI Similarity Search) to provide insightful and interactive data analysis on Singapore's HDB resale market.
    
    ## Objectives:
    1. Provide automated property valuation using LLM.
    2. Enable semantic search using FAISS and LLMs.
    
    ## Data Sources:
    The data used in this project comes from the publicly available HDB resale transactions dataset. We have enriched this dataset with additional features to enhance the LLM predictions.
    
    ## Features:
    - **Automated Property Valuation**: A feature to estimate HDB resale prices based on various input parameters.
    - **LLM-based Semantic Search**: Enables users to query the HDB resale data in natural language and returns results based on FAISS similarity search.
    
    ## How to Use:
    - **Automated Property Valuation**: 
      1. Go to the “Automated Property Valuation” page from the sidebar.
      2. Input the required fields such as Transaction Year, Town, Flat Type, and others.
      3. Click on "Predict Resale Price with LLM" to get an estimated resale price.
    
    - **LLM-based Semantic Search**:
      1. Go to the “LLM-based Semantic Search” page.
      2. Enter a query (e.g., "Show me properties in Bedok with 90 sqm flat area").
      3. The app will return the best matching results using FAISS similarity search.
    """)

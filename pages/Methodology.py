import streamlit as st

# region <--------- Streamlit App Configuration --------->
st.set_page_config(
    layout="centered",
    page_title="Methodology"
)
# endregion <--------- Streamlit App Configuration --------->


st.title("Methodology")
st.write("""
    ## Data Flow and Implementation Details:
    
    1. **Data Loading**: The HDB resale dataset is loaded from a CSV file. Each row contains information on the town, flat type, floor area, storey range, and other relevant details.
    
    2. **Embedding Generation**: We use the `SentenceTransformer` model to convert textual data (e.g., town, flat type, floor area) into numerical embeddings. These embeddings are then stored or used to create a FAISS index.
    
    3. **FAISS Index**: The FAISS index allows for efficient similarity search over the embeddings. Queries from users are also converted into embeddings and compared against the index to find the most similar entries.
    
    4. **LLM Integration**: For automated property valuation, we use a pre-trained LLM (from OpenAI) and prompt it with specific input parameters such as flat type, floor area, storey range, and town to generate price estimates.
    
    ## Flowchart:
    Below is a high-level flowchart that illustrates the process flow for both main use cases:
    1. **Automated Property Valuation**
    2. **LLM-based Semantic Search**

    """)

st.image("./data/Flowchart.jpeg")
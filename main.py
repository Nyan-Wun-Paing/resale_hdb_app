import streamlit as st
import pandas as pd
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import numpy as np
import os
from dotenv import load_dotenv
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st  
from utility import check_password  

# Check if the password is correct.  
if not check_password():  
    st.stop()

# Load environment variables from .env file
load_dotenv('.env')

# OpenAI API initialization
if load_dotenv('.env'):
   # for local development
   OPENAI_KEY = os.getenv('OPENAI_API_KEY')
else:
   OPENAI_KEY = st.secrets['OPENAI_API_KEY']


llm = OpenAI(api_key=OPENAI_KEY)

# Retry mechanism using Tenacity for handling RateLimitError
@retry(
    wait=wait_exponential(min=1, max=60),  # Exponential backoff
    stop=stop_after_attempt(10),           # Retry up to 10 times
    retry=retry_if_exception_type(RateLimitError)  # Only retry on RateLimitError
)
def run_llm_chain(chain, inputs):
    """Run the LLMChain with retries in case of RateLimitError"""
    return chain.run(inputs)

# Load HDB resale dataset
@st.cache_data
def load_data():
    df = pd.read_csv("./data/hdb_resale_data.csv")
    return df

@st.cache_data
def load_or_create_faiss_index(df):
    index_file = 'faiss_index'
    embeddings_file = 'embeddings.npy'

    # Initialize sentence transformer model locally
    model = SentenceTransformer('all-MiniLM-L6-v2')

    if os.path.exists(index_file) and os.path.exists(embeddings_file):
        st.write("Loading precomputed FAISS index and embeddings...")
        
        # Load FAISS index directly without pickle
        index = faiss.read_index(index_file)
        combined_texts = np.load(embeddings_file, allow_pickle=True)
        
    else:
        st.write("Creating FAISS index from scratch...")

        # Combine relevant data into a single text field for embedding
        df['combined_info'] = df.apply(lambda row: f"{row['town']} {row['flat_type']} {row['floor_area_sqm']} sqm, {row['flat_model']}, {row['storey_range']}", axis=1)

        # Generate embeddings for the combined info
        combined_texts = df['combined_info'].tolist()
        combined_embeddings = model.encode(combined_texts)

        # Save embeddings for future use
        np.save(embeddings_file, combined_texts)

        # Create the FAISS index
        d = combined_embeddings.shape[1]  # Dimension of the embeddings
        index = faiss.IndexFlatL2(d)  # L2 distance for similarity search
        index.add(np.array(combined_embeddings))  # Add embeddings to index

        # Save FAISS index without pickle
        faiss.write_index(index, index_file)

    return index, combined_texts



# Main function for the app
def main():
    st.title("HDB Resale Data App with LLM")

    df = load_data()
    st.subheader("Sample Data")
    st.write(df.head())

    option = st.sidebar.selectbox(
        'Choose a Use Case',
        ['Automated Property Valuation', 'LLM-based Semantic Search']
    )

    if option == 'Automated Property Valuation':
        automated_valuation_with_llm(df)

    if option == 'LLM-based Semantic Search':
        semantic_search(df)

# Automated Property Valuation Use Case with LLMChain
def automated_valuation_with_llm(df):
    st.subheader("Automated Property Valuation using LLM")

    df['floor_area_sqft'] = df['floor_area_sqm'] * 10.7639
    df['flat_age'] = 2024 - df['lease_commence_date']

    tranc_year = st.number_input("Transaction Year", min_value=1990, max_value=2024, value=2015)
    town = st.selectbox("Town", df['town'].unique())
    flat_type = st.selectbox("Flat Type", df['flat_type'].unique())
    storey_range = st.selectbox("Storey Range", df['storey_range'].unique())
    floor_area = st.number_input("Enter Floor Area (in sqft)", min_value=300, max_value=2000, step=10)
    flat_age = st.number_input("Enter Flat Age (in years)", min_value=1, max_value=99, step=1)
    mall_distance = st.number_input("Mall Nearest Distance (in meters)", min_value=0, max_value=5000, step=100)
    hawker_proximity = st.selectbox("Hawker Center Proximity", ["Within 500m", "Within 1km", "Within 2km", "More than 2km"])
    mrt_distance = st.number_input("MRT Nearest Distance (in meters)", min_value=0, max_value=5000, step=100)

    valuation_template = """
    You are an expert in Singapore HDB resale properties. I need you to help me predict the resale price of an HDB flat.
    
    Here are the details:
    - Transaction Year: {tranc_year}
    - Town: {town}
    - Flat Type: {flat_type}
    - Storey Range: {storey_range}
    - Floor Area (sqft): {floor_area}
    - Flat Age (years): {flat_age}
    - Mall Nearest Distance (meters): {mall_distance}
    - Hawker Center Proximity: {hawker_proximity}
    - MRT Nearest Distance (meters): {mrt_distance}
    
    Based on the above information, what is your estimated resale price for this flat in Singapore dollars (SGD)?
    """

    prompt = PromptTemplate(
        input_variables=["tranc_year", "town", "flat_type", "storey_range", "floor_area", "flat_age", "mall_distance", "hawker_proximity", "mrt_distance"],
        template=valuation_template
    )
    
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    if st.button("Predict Resale Price with LLM"):
        llm_input = {
            "tranc_year": tranc_year,
            "town": town,
            "flat_type": flat_type,
            "storey_range": storey_range,
            "floor_area": floor_area,
            "flat_age": flat_age,
            "mall_distance": mall_distance,
            "hawker_proximity": hawker_proximity,
            "mrt_distance": mrt_distance
        }

        predicted_price = run_llm_chain(llm_chain, llm_input)
        st.write(f"Output: {predicted_price}")

# LLM-based Semantic Search Use Case
def semantic_search(df):
    st.subheader("LLM-based Semantic Search")

    # Load or create FAISS index
    index, combined_texts = load_or_create_faiss_index(df)

    user_query = st.text_input("Ask a question about HDB resale data", "")

    if user_query:
        # Convert user query to embedding
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([user_query])

        # Perform similarity search using FAISS
        D, I = index.search(np.array(query_embedding), k=1)  # Top 5 results

        # Show results
        st.write("Top matching properties:")
        for i in I[0]:
            st.write(combined_texts[i])

if __name__ == "__main__":
    main()


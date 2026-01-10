import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from PyPDF2 import Pdfreader

pdf_file = st.file_uploader("Upload your file", type ="pdf")

#extract text

text=""

if pdf_file is not None:
    reader = Pdfreader(pdf_file)
    
    for page_text in reader.pages:
        if page_text:
            text+=page_text.extract_text()

#split text into chunks

text_spliter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function= len
)

#Save chunks
data = text_spliter.split_text(text)

if len(data) == 0:
    st.stop()

#Create embeedings

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

database = FAISS.from_texts(data,embeddings)

#User input

u_input = st.text_input("Please ask questions about PDF file")

llm = Ollama(model="llama3")

if u_input :
    search_result = databse.similarity_search(user_input)
    chain = load_qa_chain(llm,chain_type="stuff",verbose=True)

    with get_openai_callback() as cb:
        reponse = chain.run(input_documents= search_result, question=u_input)
        print(cb)

    st.write(response)

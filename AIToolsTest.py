import streamlit as st
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_community.callbacks import get_openai_callback
from PyPDF2 import PdfReader

pdf_file = st.file_uploader("Upload your file", type ="pdf")

#extract text

text=""

if pdf_file is not None:
    reader = PdfReader(pdf_file)
    
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

llm = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.3", task="conversational", huggingfacehub_api_token="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")

if u_input :
    search_result = database.similarity_search(u_input)
    chain = load_qa_chain(llm,chain_type="stuff",verbose=True)

    with get_openai_callback() as cb:
        reponse = chain.run(input_documents= search_result, question=u_input)
        print(cb)

    st.write(response)

# Streamlit Community Cloud의 내장 sqlite3과 Chroma 간 호환성 에러 발생으로 인해 pysqlite3을 사용하는 코드를 추가
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Import
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_classic import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import tempfile
import os
from streamlit_extras.buy_me_a_coffee import button
from dotenv import load_dotenv
load_dotenv()

# 제목
st.title("ChatPDF")
st.write("---")

# OpenAI 키 입력받기 
openai_key = st.text_input('OPEN_AI_API_KEY', type="password")

# 파일 업로드 
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'])
st.write("---")

# Buy me a coffee
button(username="skygudanr", floating=True, width=221)

# PDF 파일을 저장하고 저장한 PDF를 읽어서 페이지 단위로 분할하여 리턴하는 함수
def pdf_to_document(upload_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    # PDF 로더 
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split() # PDF 문서를 페이지 단위로 분할하여 로드
    return pages

# 업로드된 파일 처리 
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)

    # Splitter
    text_splitter = RecursiveCharacterTextSplitter(
        # 작은 청크 단위로 설정
        chunk_size=300, # 청크의 최대 길이
        chunk_overlap=20, # 청크 간 중복되는 영역 길이
        length_function=len, # 청크 길이 측정 함수를 len으로 설정
        is_separator_regex=False, # 구분자를 단순한 문자열로 해석
    )
    texts = text_splitter.split_documents(pages)

    # Embedding 
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        openai_api_key=openai_key
        # dimensions=1024 # 반환될 임베딩 크기 지정
    )

    # 첫 실행 이후 "Could not connect to tenant default_tenant. Are you sure it exists" 에러 발생시 클라이언트 캐시를 삭제하는 코드 추가 
    import chromadb
    chromadb.api.client.SharedSystemClient.clear_system_cache()

    # Chroma DB
    db = Chroma.from_documents(texts, embeddings_model)
    
    # User Input
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input("질문을 입력하세요")

    if st.button("질문하기"):
        with st.spinner("Wait for it..."):
            # Retriever
            llm = ChatOpenAI(temperature=0)
            retriever_from_llm = MultiQueryRetriever.from_llm(
                retriever=db.as_retriever(), llm=llm
            )

            # Prompt Template
            prompt = hub.pull("rlm/rag-prompt")


            # Generate
            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            rag_chain = (
                {
                    "context": retriever_from_llm | format_docs, 
                    "question": RunnablePassthrough() # 입력 데이터를 그대로 다음 단계로 전달하는 특별한 Runnable
                }
                | prompt
                | llm
                | StrOutputParser()
            )

            # Question
            result = rag_chain.invoke(question)
            st.write(result)


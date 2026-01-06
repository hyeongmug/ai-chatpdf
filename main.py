# Streamlit Community Cloud의 내장 sqlite3과 Chroma 간 호환성 에러 발생으로 인해 pysqlite3을 사용하는 코드를 추가
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except Exception:
    pass

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
from langchain_classic.callbacks.base import BaseCallbackHandler
from error_handler import ErrorInterceptor, safe_operation
from dotenv import load_dotenv
load_dotenv()

# 세션 상태 초기화
if 'api_key_valid' not in st.session_state: st.session_state.api_key_valid = False
if 'openai_key' not in st.session_state: st.session_state.openai_key = None
if 'db_ready' not in st.session_state: st.session_state.db_ready = False
if 'api_key_error' not in st.session_state: st.session_state.api_key_error = False

# 제목
st.title("ChatPDF")
st.write("---")

# OpenAI 키 입력
openai_key = st.text_input('OPEN_AI_API_KEY', type="password", key="api_key_input")

# 파일 업로드 (항상 표시)
uploaded_file = st.file_uploader("PDF 파일을 올려주세요!", type=['pdf'], key="pdf_uploader")
st.write("---")

# Buy me a coffee (원래 위치)
button(username="skygudanr", floating=True, width=221)

# API 키 상태에 따른 명확한 안내
if not openai_key or openai_key.strip() == "":
    st.warning("🔑 **OpenAI API 키를 입력해주세요**")
elif not st.session_state.api_key_valid:
    # 유효성 검사 시도
    try:
        from openai import OpenAI
        client = OpenAI(api_key=openai_key)
        client.models.list()
        st.session_state.api_key_valid = True
        st.session_state.openai_key = openai_key
        st.session_state.api_key_error = False
        st.success("✅ **API 키 확인 완료!**")
        st.rerun()
    except Exception as e:
        st.session_state.api_key_valid = False
        st.session_state.api_key_error = True
        ErrorInterceptor._handle_error("API_KEY_VALIDATION", e)
else:
    # 유효한 키일 때만 다음 단계 진행
    st.success("✅ API 키 정상")

openai_key = st.session_state.openai_key

# PDF 처리 함수
@safe_operation
def pdf_to_document(upload_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, upload_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(upload_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# PDF 처리 (API 키 유효 + 파일 업로드 시)
if uploaded_file is not None and st.session_state.api_key_valid and not st.session_state.db_ready:
    with st.spinner("📖 PDF 처리 중..."):
        pages = ErrorInterceptor.safe_execute(pdf_to_document, uploaded_file)
        if pages is None: st.stop()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=20, length_function=len, is_separator_regex=False
        )
        texts = text_splitter.split_documents(pages)
        
        try:
            import chromadb
            chromadb.api.client.SharedSystemClient.clear_system_cache()
        except: pass
        
        @safe_operation
        def create_vector_db():
            embeddings_model = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=openai_key)
            return Chroma.from_documents(texts, embeddings_model)
        
        st.session_state.db = ErrorInterceptor.safe_execute(create_vector_db)
        if st.session_state.db:
            st.session_state.db_ready = True
            st.success("✅ **PDF 처리 완료! 질문 시작하세요.**")

# 질문 UI
if st.session_state.get('db_ready', False):
    class StreamHandler(BaseCallbackHandler):
        def __init__(self, container, initial_text=""): 
            self.container = container
            self.text = initial_text
        
        def on_llm_new_token(self, token: str, **kwargs) -> None:
            try:
                self.text += token
                self.container.markdown(self.text)
            except: pass
    
    st.header("💬 PDF에게 질문해보세요!")
    question = st.text_input("질문을 입력하세요:")
    
    if st.button("질문하기", type="primary") and question.strip():
        with st.spinner("🤔 답변 생성 중..."):
            try:
                db = st.session_state.db
                llm = ChatOpenAI(temperature=0, openai_api_key=openai_key)
                retriever_from_llm = MultiQueryRetriever.from_llm(retriever=db.as_retriever(), llm=llm)
                prompt = hub.pull("rlm/rag-prompt")
                
                chat_box = st.empty()
                stream_handler = StreamHandler(chat_box, "**답변:** ")
                generate_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=openai_key, streaming=True, callbacks=[stream_handler])
                
                def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)
                
                rag_chain = (
                    {"context": retriever_from_llm | format_docs, "question": RunnablePassthrough()}
                    | prompt | generate_llm | StrOutputParser()
                )
                rag_chain.invoke(question)
            except Exception as e:
                ErrorInterceptor._handle_error("RAG_GENERATION", e)

# 최종 상태 안내
elif st.session_state.api_key_valid and uploaded_file:
    st.info("⏳ **PDF 처리 완료 대기 중...**")
elif st.session_state.api_key_valid:
    st.info("📄 **PDF 파일을 업로드하세요**")

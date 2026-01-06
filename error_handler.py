import streamlit as st
import logging
import traceback
from typing import Callable, Any
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorInterceptor:
    @staticmethod
    def safe_execute(func: Callable, *args, **kwargs) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            ErrorInterceptor._handle_error(func.__name__, e)
            return None
    
    @staticmethod
    def _handle_error(func_name: str, error: Exception):
        error_msg = str(error).lower()
        logger.error(f"[{func_name}] {str(error)}\n{traceback.format_exc()}")
        
        if any(keyword in error_msg for keyword in [
            "incorrect api key", "invalid_api_key", "invalid api", 
            "authenticationerror", "401", "api key provided"
        ]):
            st.error("❌ **잘못된 API 키입니다**")
        elif "pdf" in error_msg or "file" in error_msg:
            st.error("❌ **PDF 파일을 다시 업로드해주세요**")
        else:
            st.error("❌ **일시적인 오류가 발생했습니다**")

def safe_operation(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return ErrorInterceptor.safe_execute(func, *args, **kwargs)
    return wrapper

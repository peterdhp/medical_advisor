import streamlit as st
from menu_streamlit import menu
from utils_streamlit import *
import os

from menu_streamlit import menu

from langsmith import traceable

if "openai_api_key_psscode" not in st.session_state:
    st.session_state.openai_api_key_psscode = ''


os.environ["LANGCHAIN_API_KEY"]=st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"]=st.secrets['LANGCHAIN_TRACING_V2']
os.environ["LANGCHAIN_ENDPOINT"]=st.secrets['LANGCHAIN_ENDPOINT']
os.environ['LANGCHAIN_PROJECT']=st.secrets['LANGCHAIN_PROJECT']

os.environ['CO_API_KEY']=st.secrets['CO_API_KEY']

st.title("OpenAI API Key 등록")

st.session_state.openai_api_key_psscode = st.text_input('OpenAI API Key' ,type='password')

    
    
if st.session_state.openai_api_key_psscode == 'yongari':
    st.session_state.openai_api_key_psscode = st.secrets['OPENAI_API_KEY']
    
    os.environ['OPENAI_API_KEY']=st.secrets['OPENAI_API_KEY']
if not st.session_state.openai_api_key_psscode.startswith('sk-'):
    st.warning('OpenAI API key를 입력해주세요!', icon='⚠')
    
if st.session_state.openai_api_key_psscode.startswith('sk-'):
    st.switch_page('pages/patient_info.py')
    
    

    
menu()
    




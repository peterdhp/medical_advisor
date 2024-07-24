import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAI
from langchain_core.output_parsers import StrOutputParser

from langchain_community.callbacks import get_openai_callback
from menu_streamlit import menu_with_redirect

from langchain_core.pydantic_v1 import BaseModel, Field

from langchain_community.vectorstores import FAISS
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langsmith import traceable
import cohere

from operator import itemgetter

from typing import List
from langchain_core.documents import Document
from langchain_openai.embeddings import OpenAIEmbeddings
import os
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
    RunnableAssign
)


os.environ["CO_API_KEY"] = st.secrets['CO_API_KEY']

if "messages" not in st.session_state:
    st.session_state.messages = []
    
    

def format_docs_with_id(docs: List[Document]) -> str:
    #print(docs)
    formatted = [
        f"source ID : {i}\ntitle : {doc.metadata['title']}\nGuideline snippet :\n{doc.metadata['prechunk']} {doc.page_content} {doc.metadata['postchunk']}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n".join(formatted)

def retrieve_and_merge(query_list : list[str]) -> list[Document] :
    FAISS_PATH = 'faiss' 
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = FAISS.load_local(folder_path=FAISS_PATH,embeddings=embeddings,index_name ="CMC_clinical_practice_guideline", allow_dangerous_deserialization=True)
    ### k를 통해 몇개의 문서를 retrieve 해올지를 정할 수 있습니다.
    retriever= db.as_retriever(search_kwargs={'k':10, 'fetch_k':30})
    unique_docs = {}
    for query in query_list:
        result = retriever.invoke(query)
        for doc in result:
            # Using page_content as a unique key to avoid duplicates
            if doc.page_content not in unique_docs:
                unique_docs[doc.page_content] = doc
    docs_list = list(unique_docs.values())
    docs = {x.page_content: i for i, x in enumerate(docs_list)}
    rerank_input = list(docs.keys())
    
    return docs_list, rerank_input

def rerank(rerank_input,question):
    
    co = cohere.Client()
    rerank_response = co.rerank(
        query=question, documents= rerank_input, top_n=5, model="rerank-multilingual-v3.0"
    )
    
    return rerank_response

def compress_retrieve(dict):
    #print(dict["optimizedQuery"])
    query_list = dict["optimizedQuery"].query_list
    question = dict["patient_info"] + dict["question"]
    
    unique_docs, rerank_input = retrieve_and_merge(query_list)
    rerank_response = rerank(rerank_input,question)
    docs = [unique_docs[i.index] for i in rerank_response.results]
    
    return docs
    
class optimizedQuery(BaseModel):
    """list of queries that are optimized to retrieve relevant medical information in respect to the question and patient information"""

    query_list: list[str] = Field(
        description="list of queries that optimized to retrieve relevant medical information in respect to the question and patient information"

    )
    
class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The source ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class quoted_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )
    
    
def LLM_respond_Q(msg_log):
    """채팅 내용을 기반으로 질문을 함"""
        
    
    llm = ChatOpenAI(model="gpt-4o", temperature = 0) 
    
    query_system_prompt = [
    ("system", """
You are a specialized medical assistant trained to support doctors by providing accurate and relevant medical information in Korean. When given a patient information, generate a list of Korean queries optimized to retrieve the most appropriate and personalized medical information to answer the last message. 3 queries at max.

[Patient Information]
{patient_info}
""")]
    
    
    
    
    query_prompt = ChatPromptTemplate.from_messages(query_system_prompt+msg_log)
    
    
    query_llm= llm.with_structured_output(optimizedQuery)

    response_system_prompt = [("system", """You are a medical assitant. you are going to assist in medical decision. 
    Answer only based on the given context and patient information.:
    
    [Context]
    {context}
    
    [patient info]
    {patient_info}                   
    """)]

    
    response_prompt = ChatPromptTemplate.from_messages(response_system_prompt + msg_log)
    
    response_llm = llm.bind_tools(
        [quoted_answer],
        tool_choice="quoted_answer",
    )
    output_parser = JsonOutputKeyToolsParser(key_name="quoted_answer", return_single=True)

    chain = (RunnableParallel(patient_info = itemgetter('patient_info'),question = itemgetter('question'))
            .assign(optimizedQuery = itemgetter('patient_info')|query_prompt | query_llm)
            .assign(docs = {"patient_info" : itemgetter("patient_info"), "question" : itemgetter('question'), "optimizedQuery" : itemgetter('optimizedQuery')}| RunnableLambda(compress_retrieve))
            .assign(context = itemgetter('docs') | RunnableLambda(format_docs_with_id))
            .assign(answer = response_prompt | response_llm | output_parser)
            .pick(["docs","answer"]))
    
    
    output = chain.invoke({"patient_info": st.session_state.patient_info, "question" : msg_log[-1][1] })
    #print(output)
    
    return output

    
    
for message in st.session_state.messages:
    role = '🩺' if message[0] == 'ai' else message[0]
    with st.chat_message(role):
        st.markdown(message[1])
     
def format_citation_to_text(citations) :   
    text = ''
    for citation in citations :
        text = text + citation.source_title + '\n' + citation.quote + '\n\n'
        
    return text
        
toc = {
    3: '순환기내과 : 순환기 응급 질환',
    8: '순환기내과 : 고혈압',
    17: '순환기내과 : 고지혈증',
    23: '순환기내과 : 심부전',
    35: '순환기내과 : 심근증',
    40: '순환기내과 : 심장판막 질환',
    46: '순환기내과 : 심혈관 질환',
    66: '순환기내과 : 부정맥',
    76: '순환기내과 : 말초혈관질환과 폐색전증',
    83: '순환기내과 : 수술 전심장평가',
    93: '호흡기내과 : 호흡기 증상별 접근과 기능검사',
    112: '호흡기내과 : 기관지천식',
    119: '호흡기내과 : 만성폐쇄성폐질환',
    124: '호흡기내과 : 폐렴',
    138: '호흡기내과 : 결핵',
    149: '호흡기내과 : 간질성 폐질환',
    165: '호흡기내과 : 기타호흡기질환',
    184: '호흡기내과 : 중환자의학총론',
    198: '호흡기내과 : 호흡부전',
    205: '호흡기내과 : 알레르기검사와 질환별 접근',
    219: '신장내과 : 신장질환의 진단',
    229: '신장내과 : 전해질 및 산염기애',
    251: '신장내과 : 급성신장손상',
    265: '신장내과 : 만성신장병（투석전）',
    276: '신장내과 : 신대체요법',
    308: '신장내과 : 주요신장질환',
    342: '신장내과 : 임신과신장',
    349: '신장내과 : 신장과약물',
    361: '위장관 및 췌담도 : 위장관내시경검사',
    369: '위장관 및 췌담도 : 위장관출혈',
    377: '위장관 및 췌담도 : 식도염, 식도 운동질환 및 식도암',
    386: '위장관 및 췌담도 : 위염, 헬리코박터, 소화성궤양',
    396: '위장관 및 췌담도 : 위용종 및 위암',
    406: '위장관 및 췌담도 : 설사, 변비 및 과민성장증후군',
    425: '위장관 및 췌담도 : 염증성 장질환, 기타 장질환',
    443: '위장관 및 췌담도 : 대장용종 및 대장암',
    448: '위장관 및 췌담도 : 췌장염 및 췌장낭종성질환',
    459: '위장관 및 췌담도 : 담낭 및 담도질환',
    469: '간질환 : 간질환환자접근법',
    478: '간질환 : A형 간염',
    481: '간질환 : B형 간염',
    488: '간질환 : C형 간염',
    493: '간질환 : 알코올 간질환 및 비알코올 지방간질환',
    497: '간질환 : 독성간염',
    500: '간질환 : 간경변증 및 합병증',
    514: '간질환 : 급성 간부전과 간이식',
    518: '간질환 : 간암과 간의 양성종양',
    528: '간질환 : 자가면역성간염 및 대사성 간질환, 기타 간질환',
    541: '혈액내과 : 혈액질환 환자의 평가 및 처치',
    551: '혈액내과 : 빈혈',
    559: '혈액내과 : 골수부전증후군',
    570: '혈액내과 : 급성백혈병',
    581: '혈액내과 : 만성백혈병',
    589: '혈액내과 : 형질세포질환',
    597: '혈액내과 : 악성 림프종',
    610: '혈액내과 : 골수증식성 질환',
    617: '혈액내과 : 출혈과 지혈',
    628: '혈액내과 : 혈액 환자의 수혈 요법',
    636: '혈액내과 : 조혈모세포이식',
    645: '종양내과 : 종양환자의 접근 및 임상시험',
    648: '종양내과 : 항암화학요법의 원칙 및 부작용',
    661: '종양내과 : 폐암',
    674: '종양내과 : 유방암',
    682: '종양내과 : 위암',
    688: '종양내과 : 대장암',
    693: '종양내과 : 기타암 (두경부암, 식도암, 췌/담도암, 육종/GIST)',
    703: '종양내과 : 원발 병소불명암',
    706: '종양내과 : 지지요법',
    715: '종양내과 : 종양학응급',
    725: '내분비내과 : 뇌하수체질환',
    742: '내분비내과 : 갑상선 질환',
    769: '내분비내과 : 당뇨병',
    788: '내분비내과 : 저혈당증',
    794: '내분비내과 : 부신 질환',
    808: '내분비내과 : 칼슘 및 골대사질환',
    827: '내분비내과 : 기타대사질환',
    836: '내분비내과 : 기타내분비질환',
    849: '류마티스내과 : 근골격계증상 접근법',
    854: '류마티스내과 : 류마티스질환진단검사',
    858: '류마티스내과 : 류마티스약물',
    865: '류마티스내과 : 전신홍반루푸스',
    874: '류마티스내과 : 항인지질증후군',
    876: '류마티스내과 : 류마티스관절염',
    881: '류마티스내과 : 전신경화증',
    888: '류마티스내과 : 쇼그렌증후군',
    891: '류마티스내과 : 강직성척추염',
    899: '류마티스내과 : 혈관염과 베체트병',
    909: '류마티스내과 : 염증성근병증',
    915: '류마티스내과 : 골관절염',
    920: '류마티스내과 : 통풍',
    924: '류마티스내과 : 섬유근통',
    931: '감염내과 : 감염질환의 진단',
    938: '감염내과 : 감염질환치료',
    975: '감염내과 : 주요 감염질환',
    1055: '감염내과 : 감염질환의 예방'
}
        
def get_key_from_value(d, value):
    for k, v in d.items():
        if v == value:
            return k
    return None       
        
if userinput := st.chat_input("메시지를 입력해주세요"):
    # Add user message to chat history
    st.session_state.messages.append(("human", userinput))
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(userinput)
    with st.chat_message("🩺"):
        output = LLM_respond_Q(st.session_state.messages)
        #print(output)
        answer = output['answer'][0]['answer']
        citations = output['answer'][0]['citations']
        docs = output['docs']
        st.markdown(answer)
    st.session_state.messages.append(("ai", answer))
    for i, citation in enumerate(citations):
        with st.expander(citation['quote']):
            st.markdown(docs[citation['source_id']].metadata['title']+ "        CMC 내과진료지침서 "+ str(get_key_from_value(toc, docs[citation['source_id']].metadata['title']))+"쪽")
            st.markdown(f"{docs[citation['source_id']].metadata['prechunk']} {docs[citation['source_id']].page_content} {docs[citation['source_id']].metadata['postchunk']}")
            
            
    


    
    
menu_with_redirect()

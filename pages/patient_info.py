import streamlit as st
from menu_streamlit import menu_with_redirect



    
if "patient_info" not in st.session_state:
    st.session_state.patient_info = ''
if "_patient_info" not in st.session_state:
    st.session_state._patient_info = ''


def submit():
    st.session_state.patient_info=st.session_state._patient_info
    


st.title("환자 정보 입력")

#st.text(f"ESR, O반응 단느색(CRP),  procalcitonin/n(4) 소변검사/n(5) 혈액 또는 기타 배양검사 : 객담, 소변 등/n(6) 뇌척수액이나  비정상적인  체액(흉수, 복수, 관절액 )의 검사 및 배양/n(7) 대변검사 (잠혈반응 , 백혈구 , 기생충  등)/n(8) 말초혈액도말검사/n4) 영상검사/n초기에는  흉부나  복부의  단순 사진으로  폐렴, 위장관  감염, 간 • 비장 비대 등을 확인하고 , 필요시  복 /n부/흉부 컴퓨터단층촬영을  실시하여  농양, 종괴, 림프절  종대, 간 • 비장 비대 등을 확인/n5) 발열 치료의 원칙/n관습적 해열제는  옳지 않음. 원인 감별을  위한 노력이  우선. 단, 다음과  같은 상황에서는  발열을  조절 /n해볼수있음/n(1) 발열로  인한 산소 요구량의  증가가  문제가  되는 경우: 심장질환 , 뇌혈관장애 , 호흡부전/n(2) 간질 환자/n(3) 열성 경련의  병력이  있는 소아/n 제 1-2절 배양검사  결과 해석/n| 三 링크 ： 卜«0%)109.=/ 巾09내0/130153215329/nI . 배양검사  해석에 필요한  사항들/n1. 배양된 균이 그람양성인지 , 음성인지  파악/n1) 일단 그람염색  양성, 그람염색  음성 여부가  확인되면  사용할  필요가  없는 항생제들은  고려 대상에서  /n제외시킬  수 있기 때문에  그만큼  판단이  더 용이해짐/n2) 그람양성균이  나온다면  aztreonam 이나 aminoglycoside  (gentamicin,  amikacin  등), fluoroquinolone  /n(ciprofloxacin,  ofloxacin  등; 단, levofloxacin,  moxffloxacin,  gatifloxadn  등의 신세대  fluoroquinolone^  /n예외) 계통의  약제들은  일단 제외해  놓고 다른 약제들을  확인/n3) 그람음성균이  나온다면  vancomycin 이나 tekxplanin 은 고려할  필요가  없음/n2. 어떤 검체에서  나온 것인지 확인/n1) 인체는  전반적으로  무균 상태가  아니기 때문에  세균 배양에서  균이 자랐다고  해서 반드시  감염의  원 /n인 균이라고  속단해서는  안됨/n2) 정상적으로는  무균 상태이어야만  하는 검체, 즉 혈액이나  뇌척수액에서  균이 배양되었다면  의미가  /n있다고  판단해야  함/n3) 객담이나  분변에서  배양되는  세균은  원인균으로  속단해서는  안 됨. 객담 배양은  대개 상기도의  세균 /n총이, 분변 배양은  장내 세균총이  반영된  경우가  많기 때문/n3. 배양된 세균과  환자의 임상 증상이 일치하는지  확인/n세균이  배양되었다는  것이 곧 감염을  의미하는  것은 아니며 , 단순한  cdoni=bn 이나 검체 채취 과정에  9^/n 스/n 서의 오염(contamination)  가능성을  더불어 고려해야  함. 이를 위해 해당 환자의  임상 증상에  대한 사전 정 /n보를 파악하고  있어야  함/n4. 배양된 균종에 따라 어떤 항생제를  사용할지  판단/n검사 결과를  해석할  때, 검사된  모든 약제들의  감수성 여부를  일일이  다 확인할  필요까지는  없으며 , 각 균 /n종별로  중요한  항생제  몇 가지만  중점적으로  내성 여부를  확인하는  것으로  충분. 이를 위해서는  임상적으로  /n중요한  균 종류에  따라 어떤 항생제를  우선적으로  선택해야  하는지에  대한 기본 개념을  알고 있어야  함/n1) 그람양성균인  경우는  Staphylococci  Streptococci 로 나눠서  판단/nStaphylococd 는 methicillin  내성이면  vancomycin 이나 teicoplanin  같은 glycopeptide 를, 아닌 경우에  /n는 -lactam  계통이면  충분함 . 병원 밖에서 걸린 경우라면  methicillin  감수성인  경우가  많.?므루 0 /n-lactam을 주로 사용.")

with st.form(key='demographics_form'):
    
    st.text_area('환자 관련 요약 정보를 예시와 같이 입력해주세요. 환자 정보 등록을 생략할 수 있습니다.', placeholder="""뇌경색증 후유증, 2형 당뇨병, 원발성 고혈압, 전립선증식증, 기타행동장애, 변비
CC: 혈뇨+발열
Blood glucose 185
110/70-82-36.1

복용약: 
프리살탄정 - Olmesartan medoxomil (ARB)
타루날캡슐 - 전립선비대증
리스돈정 - Risperidone
마그밀 - magnesium hydroxide
티아민염산염정 - Thiamine hydrochloride (비타민 B1)
훼로맥스액 - Ferric hydroxide (철 결핍성 빈혈)
파모티딘 - H2 receptor antagonist
삐콤정 - B1, B2, B6, C 보급
아목시실 - anti biotics
애니디핀정 - Amlodipine
암브록솔 - 진해거담제
뮤코스텐캡슐 - 진해거담제
코데날정 - 진해거담제""",key='_patient_info', height=400)
    
    if st.form_submit_button("등록하기",on_click = submit):
        st.switch_page('pages/chat_assistant.py')
        
menu_with_redirect()
        

    




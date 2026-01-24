import uuid
import json
import requests
import streamlit as st


if "thred_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for content in st.session_state.chat_history:
    with st.chat_message(content["role"]):
        st.markdown(content['message'])    

if prompt := st.chat_input("메시지를 입력하세요."):
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.chat_history.append({"role": "user", "message": prompt})   
    with st.chat_message("ai"):                
        params = {
            "thread_id": st.session_state.thread_id,
            "query": prompt
        }
        placeholder = st.empty()
        full_text = ""
        with requests.get("http://localhost:8000/chat/streaming?", params=params, stream=True) as response:
            if response.status_code != 200:
                full_text = f"Received status code {response.status_code}"
                placeholder.error(full_text)
            else:
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if "DONE" in chunk:
                        break
                    for obj in chunk.split("data: "):
                        if obj:
                            print(obj)
                            obj_dict = json.loads(obj)
                            full_text += obj_dict['content']
                            placeholder.markdown(full_text + "▌")
                placeholder.markdown(full_text)
        st.session_state.chat_history.append({"role": "ai", "message": full_text})
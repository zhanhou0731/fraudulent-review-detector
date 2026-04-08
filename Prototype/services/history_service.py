import streamlit as st
from datetime import datetime

class HistoryService:
    def __init__(self):
        if 'history_dict' not in st.session_state:
            st.session_state.history_dict = {}
            
        if 'batch_history' not in st.session_state:
            st.session_state.batch_history = []

    def add_record(self, text, new_results):
        clean_key = text.strip()
        history = st.session_state.history_dict
        
        if clean_key in history:
            record = history[clean_key]

            record['results'].update(new_results)
            record['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            del history[clean_key]
            history[clean_key] = record
            
        else:
            history[clean_key] = {
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'text': text,
                'results': new_results
            }

    def get_history(self):
        return list(st.session_state.history_dict.values())[::-1]

    def clear_history(self):
        st.session_state.history_dict = {}


    def add_batch_record(self, filename, df, text_col):
        record = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'filename': filename,
            'count': len(df),
            'df': df,
            'text_col': text_col
        }

        st.session_state.batch_history.insert(0, record)
        
        if len(st.session_state.batch_history) > 5:
            st.session_state.batch_history.pop() 

    def get_batch_history(self):
        return st.session_state.batch_history

    def clear_batch_history(self):
        st.session_state.batch_history = []
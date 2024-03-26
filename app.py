import streamlit as st

import os

from loadData import DataLoader
from modelLoader import HyBridRecommender

st.write("Hello There!")
st.text_input("Nhập mã người dùng: ", key="u_id")
st.text_input("Nhập mã sản phẩm: ", key="i_id")
clicked = st.button("Đề xuất!")

# Tìm đường dẫn thư mục con và thư mục chính
base_dir = os.path.abspath(os.getcwd())

CBFDataPath = "\\Source_code\\Dữ liệu đã được xử lý\\CBF_data.csv"
CFDataPath = "\\Source_code\\Dữ liệu đã được xử lý\\CF_data.csv"

dataLoader = DataLoader(CBFDataPath, CFDataPath)
meta_data = dataLoader.meta_data
CF_data = dataLoader.CF_data
CBF_trainset = dataLoader.getCBFTrainSet()

hybridSystem = HyBridRecommender(meta_data, CF_data, CBF_trainset)

top_n = 10

if clicked:
    st.write("Khuyến nghị cho người dùng mã số " + st.session_state.u_id + " với mã sản phẩm " + st.session_state.i_id + " là: \n")
    st.write(hybridSystem.get_hybrid_recommendations(st.session_state.u_id, st.session_state.i_id, top_n))

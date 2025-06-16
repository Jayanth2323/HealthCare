from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.add_font("DejaVuSans", "", "./fonts/DejaVuSans.ttf", uni=True)
pdf.set_font("DejaVuSans", size=12)
pdf.multi_cell(0, 10, "测试 Unicode 測試 😊")
pdf.output("test_unicode.pdf")


from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(200, 10, txt="Hello Mawa!", ln=True)
pdf.output("test_output.pdf")

import sys, importlib.util
import streamlit as st
spec = importlib.util.find_spec("fpdf")
st.write("fpdf module spec:", spec)
if spec:
    st.write("fpdf file location:", spec.origin)
else:
    st.write("fpdf module not found.")
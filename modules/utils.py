import google.generativeai as genai
import shap
import streamlit.components.v1 as components
from fpdf import FPDF

def configure_genai(api_key):
    genai.configure(api_key=api_key)

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 500, scrolling=True)

def generate_pdf_report(health_summary, ai_response, path="./data/health_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, "AI Healthcare Summary Report", align="C")
    pdf.ln()
    pdf.multi_cell(0, 10, health_summary)
    pdf.ln()
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(0, 10, "Gemini's Treatment Recommendations:", ln=True)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, ai_response)
    pdf.output(path)
    return path

def generate_recommendation(pred_label):
    return {
        0: "✅ Low Risk\nMaintain a healthy lifestyle. Annual check-ups recommended.",
        1: "⚠️ Medium Risk\nIncrease physical activity and consult your doctor.",
        2: "🚨 High Risk\nImmediate medical attention advised.",
    }.get(pred_label, "❓ No recommendation available.")

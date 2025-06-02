import os
import shap
from fpdf import FPDF
import re
import urllib.request
import google.generativeai as genai
import streamlit.components.v1 as components

# === 1. Configure Gemini API Key ===
def configure_genai(api_key):
    genai.configure(api_key=api_key)

# === 2. Render SHAP Waterfall Plot in Streamlit ===
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 500, scrolling=True)

# === 3. Download and Setup DejaVu Font (if not present) ===
def ensure_font():
    font_dir = "fonts"
    font_path = os.path.join(font_dir, "DejaVuSans.ttf")
    if not os.path.exists(font_path):
        os.makedirs(font_dir, exist_ok=True)
        font_url = "https://github.com/dejavu-fonts/dejavu-fonts/blob/master/ttf/DejaVuSans.ttf?raw=true"
        try:
            urllib.request.urlretrieve(font_url, font_path)
        except Exception as e:
            raise RuntimeError(f"Font download failed: {e}")
    return font_path

# === 4. Custom PDF class supporting Unicode ===
class UnicodePDF(FPDF):
    def __init__(self):
        super().__init__()
        font_path = ensure_font()
        self.add_font("DejaVu", "", font_path, uni=True)
        self.set_font("DejaVu", size=12)

# === 5. Generate Unicode-safe PDF Report ===
def generate_pdf_report(health_summary, ai_response, path="./data/health_report.pdf"):
    pdf = UnicodePDF()
    pdf.add_page()

    pdf.multi_cell(0, 10, "🧠 AI Healthcare Summary Report", align="C")
    pdf.ln()
    pdf.multi_cell(0, 10, health_summary)
    pdf.ln()
    pdf.set_font("DejaVu", style='B', size=12)
    pdf.cell(0, 10, "Gemini's Treatment Recommendations:", ln=True)
    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(0, 10, ai_response)

    pdf.output(path)
    return path

# === 6. Map Risk Level to Human-Friendly Recommendation ===
def generate_recommendation(pred_label):
    return {
        0: "✅ Low Risk\nMaintain a healthy lifestyle. Annual check-ups recommended.",
        1: "⚠️ Medium Risk\nIncrease physical activity and consult your doctor.",
        2: "🚨 High Risk\nImmediate medical attention advised.",
    }.get(pred_label, "❓ No recommendation available.")

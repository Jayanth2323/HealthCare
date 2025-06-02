import os
import re
import shap
from fpdf import FPDF
import google.generativeai as genai
import streamlit.components.v1 as components

# === Configure Gemini AI ===
def configure_genai(api_key):
    genai.configure(api_key=api_key)

# === Render SHAP Waterfall Plot in Streamlit ===
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height or 500, scrolling=True)

# === Custom Unicode Font-Enabled PDF Class ===
class UnicodePDF(FPDF):
    def __init__(self):
        super().__init__()
        font_dir = os.path.join("fonts")
        font_path = os.path.join(font_dir, "DejaVuSans.ttf")
        if not os.path.exists(font_path):
            raise FileNotFoundError("DejaVuSans.ttf not found in fonts/ directory. Please add it.")
        self.add_font("DejaVu", "", font_path, uni=True)
        self.set_font("DejaVu", size=12)

# === PDF Report Generator using Unicode-Compatible Font ===
def generate_pdf_report(health_summary, ai_response, path="./data/health_report.pdf"):
    pdf = UnicodePDF()
    pdf.add_page()

    # Clean text by replacing emojis with text equivalents
    def clean_text(text):
        emoji_map = {
            "✅": "[OK]",
            "⚠️": "[WARNING]",
            "🚨": "[ALERT]",
            "🧠": "",
            "❌": "[ERROR]",
            "❓": "[UNKNOWN]",
            "💡": "[TIP]",
            "📅": "[CALENDAR]",
            "💸": "[MONEY]",
            "⏳": "[TIME]"
        }
        for emoji, replacement in emoji_map.items():
            text = text.replace(emoji, replacement)
        return text

    health_summary = clean_text(health_summary)
    ai_response = clean_text(ai_response)

    pdf.set_font("DejaVu", size=14)
    pdf.multi_cell(0, 10, "AI Healthcare Summary Report", align="C")

    pdf.ln()
    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(0, 10, health_summary)

    pdf.ln()
    pdf.set_font("DejaVu", style='B', size=12)
    pdf.cell(0, 10, "Gemini's Treatment Recommendations:", ln=True)

    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(0, 10, ai_response)

    # Save the PDF with proper encoding handling
    try:
        pdf_output = pdf.output(dest='S')
        with open(path, "wb") as f:
            f.write(pdf_output.encode('latin1', errors='replace'))
        return path
    except Exception as e:
        raise Exception(f"Failed to generate PDF: {str(e)}")

# === Risk-to-Recommendation Mapper ===
def generate_recommendation(pred_label):
    recommendation_map = {
        0: "[OK] Low Risk\nMaintain a healthy lifestyle. Annual check-ups recommended.",
        1: "[WARNING] Medium Risk\nIncrease physical activity and consult your doctor.",
        2: "[ALERT] High Risk\nImmediate medical attention advised.",
    }
    return recommendation_map.get(pred_label, "[UNKNOWN] No recommendation available.")
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
        # Ensure fonts directory exists
        font_dir = os.path.join("fonts")
        os.makedirs(font_dir, exist_ok=True)
        
        # Download DejaVu font if not present
        font_path = os.path.join(font_dir, "DejaVuSans.ttf")
        if not os.path.exists(font_path):
            try:
                import urllib.request
                font_url = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip"
                zip_path = os.path.join(font_dir, "dejavu.zip")
                urllib.request.urlretrieve(font_url, zip_path)
                
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extract("dejavu-fonts-ttf-2.37/ttf/DejaVuSans.ttf", font_dir)
                    os.rename(
                        os.path.join(font_dir, "dejavu-fonts-ttf-2.37", "ttf", "DejaVuSans.ttf"),
                        font_path
                    )
                os.remove(zip_path)
            except Exception as e:
                raise FileNotFoundError(f"Could not download DejaVu font: {str(e)}")

        self.add_font("DejaVu", "", font_path, uni=True)
        self.set_font("DejaVu", size=12)

# === Text Cleaning Function ===
def clean_text(text):
    """Replace all emojis and special characters with text equivalents"""
    emoji_map = {
        "✅": "[OK]","⚠️": "[WARNING]",
        "🚨": "[ALERT]","🧠": "[BRAIN]",
        "❌": "[ERROR]","❓": "[UNKNOWN]",
        "💡": "[TIP]","📅": "[CALENDAR]",
        "💸": "[MONEY]","⏳": "[TIME]",
        "💬": "[CHAT]","🚀": "[ROCKET]",
        "📄": "[DOCUMENT]","🔄": "[REFRESH]",
        "🔍": "[SEARCH]","📊": "[CHART]",
        "📈": "[GRAPH]","ℹ️": "[INFO]",
        "📝": "[WORD]","📑": "[NOTE]",
        "🧬": "[DNA]","🤖": "[ROBOT]",
        "📥": "[DOWNLOAD]","🔬": "[SCIENCE]",
        "📂": "[FOLDER]","👥": "[GENDER]",
        "🎂": "[BIRTHDAY]","🩸": "[BLOOD]",
        "💓": "[HEART]","🧪": "[CHEMISTRY]",
        "🩺": "[HEALTH]","🚬": "[SMOKING]",
        "🏃": "[RUNNING]","👈": "[LEFT]","👉": "[RIGHT]",        
    }
    for emoji, replacement in emoji_map.items():
        text = text.replace(emoji, replacement)
    return text

# === PDF Report Generator ===
def generate_pdf_report(health_summary, ai_response, path="./data/health_report.pdf"):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    pdf = UnicodePDF()
    pdf.add_page()

    # Clean input texts
    health_summary = clean_text(health_summary)
    ai_response = clean_text(ai_response)

    # Set document properties
    pdf.set_title("AI Healthcare Summary Report")
    pdf.set_author("Healthcare AI System")

    # Add content
    pdf.set_font("DejaVu", size=16)
    pdf.cell(0, 10, "AI Healthcare Summary Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(0, 8, health_summary)
    pdf.ln(10)

    pdf.set_font("DejaVu", style='B', size=12)
    pdf.cell(0, 10, "Gemini's Treatment Recommendations:", ln=True)
    pdf.set_font("DejaVu", size=12)
    pdf.multi_cell(0, 8, ai_response)

    # Generate and save PDF
    try:
        pdf_output = pdf.output(dest='S').encode('latin1', errors='replace')
        with open(path, "wb") as f:
            f.write(pdf_output)
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
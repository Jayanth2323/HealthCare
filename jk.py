def generate_pdf_report(health_summary: str, ai_response: str) -> str:
    """
    Generates a PDF report with the provided health_summary and AI recommendations.
    Uses DejaVuSans.ttf for Unicode support. Returns the path to the generated PDF file.
    """
    try:
        font_path = bootstrap_font()
    except RuntimeError as e:
        st.error(f"Could not prepare Unicode font: {e}")
        font_path = None

    pdf = FPDF()
    pdf.add_page()

    # Font setup
    if font_path:
        try:
            pdf.add_font("DejaVu", "", font_path, uni=True)
            pdf.set_font("DejaVu", size=12)
        except Exception as font_exc:
            st.warning(f"⚠️ Could not add DejaVu TTF font: {font_exc}. Falling back to Arial.")
            pdf.set_font("Arial", size=12)
    else:
        pdf.set_font("Arial", size=12)

    # Report Content
    pdf.multi_cell(0, 10, "AI Healthcare Summary Report", align="C")
    pdf.ln()
    pdf.multi_cell(0, 10, health_summary.strip())
    pdf.ln()

    # Recommendations Section
    try:
        pdf.set_font("DejaVu", "B", size=12)
    except Exception:
        pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Gemini's Treatment Recommendations:", ln=True)

    try:
        pdf.set_font("DejaVu", size=12)
    except Exception:
        pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, ai_response.strip())

    # Save File
    out_dir = os.path.join(BASE_DIR, "data")
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"health_report_{uuid.uuid4().hex}.pdf")

    try:
        pdf.output(filename)
    except Exception as export_exc:
        st.error(f"❌ Unable to save PDF report: {export_exc}")
        return ""

    return filename

def generate_pdf_report(health_summary: str, ai_response: str) -> str:
    try:
        font_path = bootstrap_font()
    except RuntimeError as e:
        st.error(f"Could not prepare Unicode font: {e}")
        font_path = None

    pdf = FPDF()
    pdf.add_page()

    if font_path:
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
    else:
        pdf.set_font("Arial", size=12)

    # … rest of your content …

    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    filename = os.path.join(BASE_DIR, "data", f"health_report_{uuid.uuid4().hex}.pdf")

    try:
        pdf.output(filename)
    except Exception as export_exc:
        st.error(f"Unable to save PDF report: {export_exc}")
        return ""

    return filename

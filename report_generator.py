import os
import urllib.request
import logging
import streamlit as st # for st.error / st.warning (assuming a Streamlit environment)

# Directory/filename constants for the DejaVuSans TTF file
FONT_DIR = "fonts"
FONT_NAME = "DejaVuSans.ttf"
FONT_PATH = os.path.join(FONT_DIR, FONT_NAME)

# Updated raw URL: Using a known working direct link for DejaVuSans.ttf
# The previous URL was returning a 404. This new URL is from a GitHub repository
# that includes the font file directly.
RAW_URL = (
    "https://github.com/telerik/kendo-keynote-resources/"
    "raw/master/q2-2015/foundation_bootstrap_demos/kendo-ui/styles/fonts/DejaVu/DejaVuSans.ttf"
)

def _is_valid_ttf(path: str) -> bool:
    """
    Quick sanity-check: TrueType files typically begin with 0x00010000 (uint32) or b'true'.
    Returns True if the file exists and starts with a valid TTF header.
    """
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as fh:
            tag = fh.read(4)
        return tag in (b"\x00\x01\x00\x00", b"true")
    except Exception as e:
        logging.error(f"Error checking TTF validity for {path}: {e}")
        return False

def bootstrap_font() -> str:
    """
    Ensure that FONT_PATH points to a valid DejaVuSans.ttf.
    If missing or invalid, download from RAW_URL.
    Raises RuntimeError if download/validation fails.
    Returns the path to a valid TTF file.
    """
    # Ensure the font directory exists before attempting to download
    os.makedirs(FONT_DIR, exist_ok=True)

    # If the file doesn’t exist or fails the header check, attempt to download
    if not _is_valid_ttf(FONT_PATH):
        logging.info(f"DejaVuSans.ttf not found or invalid at {FONT_PATH}. Attempting to download...")
        try:
            urllib.request.urlretrieve(RAW_URL, FONT_PATH)
            logging.info(f"Downloaded DejaVuSans.ttf to {FONT_PATH}")
        except Exception as download_exc:
            raise RuntimeError(f"Font download failed from {RAW_URL} → {download_exc}") from download_exc

        # Re-check after download
        if not _is_valid_ttf(FONT_PATH):
            raise RuntimeError("Downloaded file is not a valid TrueType font.")
    else:
        logging.info(f"DejaVuSans.ttf already exists and is valid at {FONT_PATH}")

    return FONT_PATH

def generate_pdf_report(health_summary: str, ai_response: str) -> str:
    """
    Generates a PDF report containing the provided health_summary and AI recommendations.
    Uses DejaVuSans.ttf (downloaded via bootstrap_font) to support Unicode.
    Returns the path to the generated PDF, or an empty string if generation failed.
    """
    # Import FPDF here to ensure it's available when this function is called
    try:
        from fpdf import FPDF
    except ImportError:
        st.error("❌ The 'fpdf' library is not installed. Please install it with 'pip install fpdf'.")
        return ""

    # Step 1: Ensure a valid TTF is available
    try:
        font_path = bootstrap_font()
    except Exception as font_boot_exc:
        # Using st.error for Streamlit integration
        st.error(f"❌ Unicode font bootstrap failed: {font_boot_exc}")
        logging.error(f"Unicode font bootstrap failed: {font_boot_exc}")
        return ""

    # Step 2: Initialize FPDF and add a page
    pdf = FPDF()
    pdf.add_page()

    # Step 3: Attempt to register the DejaVu font
    try:
        # Note: add_font expects the font family name, style, and path.
        # uni=True is crucial for Unicode support.
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
        logging.info(f"Successfully loaded DejaVu font from {font_path}")
    except Exception as font_exc:
        # Fallback if DejaVu font fails to load
        st.warning(f"⚠️ Could not add DejaVu TTF font: {font_exc}\nFalling back to Arial.")
        logging.warning(f"Could not add DejaVu TTF font: {font_exc}. Falling back to Arial.")
        pdf.set_font("Arial", size=12)

    # Step 4: Write the report title and the health summary
    # Using multi_cell for text that might wrap
    pdf.multi_cell(0, 10, "AI Healthcare Summary Report", align="C")
    pdf.ln(15) # Add more space after title

    pdf.set_font(pdf.font_family, size=12) # Reapply font in case it changed
    pdf.multi_cell(0, 8, health_summary.strip())
    pdf.ln(10)

    # Step 5: Write the Gemini recommendations section
    try:
        pdf.set_font("DejaVu", "B", size=12)
    except:
        pdf.set_font("Arial", "B", size=12)
    pdf.cell(0, 10, "Gemini's Treatment Recommendations:", ln=True)
    pdf.ln(5)

    try:
        pdf.set_font("DejaVu", size=12)
    except:
        pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 8, ai_response.strip())

    # Step 6: Save the PDF to disk
    os.makedirs("data", exist_ok=True)
    outfile = os.path.join("data", "health_report.pdf")

    try:
        pdf.output(outfile)
        logging.info(f"PDF report successfully saved to {outfile}")
    except Exception as export_exc:
        st.error(f"❌ Unable to save PDF report: {export_exc}")
        logging.error(f"Unable to save PDF report: {export_exc}")
        return ""

    return outfile

# Example Usage (for testing the PDF generation outside a Streamlit app)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) # Configure logging

    # Mock st.error and st.warning for standalone execution if not in Streamlit
    if 'st' not in globals() or not hasattr(st, 'error'):
        class MockStreamlit:
            def error(self, message):
                print(f"ERROR: {message}")
            def warning(self, message):
                print(f"WARNING: {message}")
        st = MockStreamlit() # Assign the mock object to st

    sample_health_summary = (
        "Patient exhibits symptoms of seasonal allergies including sneezing, runny nose, "
        "and itchy eyes. No fever or other signs of infection observed. "
        "Past medical history includes mild asthma, well-controlled with inhaler."
    )
    sample_ai_response = (
        "Based on the reported symptoms and medical history, it is recommended to:\n"
        "1. Continue current asthma medication regimen.\n"
        "2. Consider over-the-counter antihistamines (e.g., Cetirizine or Loratadine) for allergy relief.\n"
        "3. Advise regular saline nasal rinses to help clear allergens.\n"
        "4. Suggest avoiding known allergens if possible.\n"
        "5. Recommend consulting a healthcare professional if symptoms worsen or do not improve with self-care measures."
    )

    pdf_file = generate_pdf_report(sample_health_summary, sample_ai_response)

    if pdf_file:
        print(f"\nPDF report generated successfully at: {os.path.abspath(pdf_file)}")
        print("Please check the 'data' directory for 'health_report.pdf'.")
    else:
        print("\nFailed to generate PDF report.")


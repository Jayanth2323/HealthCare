from fpdf import FPDF
pdf = FPDF()
pdf.add_page()
pdf.add_font("DejaVuSans", "", "./fonts/DejaVuSans.ttf", uni=True)
pdf.set_font("DejaVuSans", size=12)
pdf.multi_cell(0, 10, "测试 Unicode 測試 😊")
pdf.output("test_unicode.pdf")

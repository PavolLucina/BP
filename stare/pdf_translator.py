import fitz  # PyMuPDF
from deep_translator import GoogleTranslator
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from io import BytesIO
from reportlab.lib.pagesizes import letter


def translate_text(text, target_language='sk'):
    """Translate text to the target language."""
    return GoogleTranslator(source='auto', target=target_language).translate(text)


def translate_pdf(input_path, output_path, target_language='sk'):
    """Translate a PDF by overlaying translated text."""
    # Register a Slovak-compatible font
    pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))

    # Open the input PDF
    pdf_document = fitz.open(input_path)

    # Iterate through pages
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        # Create a canvas for overlay
        packet = BytesIO()
        c = canvas.Canvas(packet, pagesize=letter)
        c.setFont('DejaVuSans', 10)

        # Iterate through text blocks on the page
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text = block[:5]  # Extract coordinates and text
            print(f"Processing text block: '{text}' on page {page_num + 1}")

            # Translate the text
            translated_text = translate_text(text, target_language)

            # Add a white rectangle to cover the original text
            c.setFillColorRGB(1, 1, 1)  # White color
            c.rect(x0, page.rect.height - y1, x1 - x0, y1 - y0, fill=1, stroke=0)

            # Add translated text
            c.setFillColorRGB(0, 0, 0)  # Black color
            c.drawString(x0, page.rect.height - y0 + 10, translated_text)

        c.save()
        packet.seek(0)

        # Overlay the canvas onto the original page
        overlay_pdf = fitz.open("pdf", packet.getvalue())
        page.insert_font(overlay_pdf)

    # Save the translated PDF
    pdf_document.save(output_path)
    pdf_document.close()




input_pdf_path = "53445334.pdf"
output_pdf_path = "53445334_sk.pdf"

# Translate the PDF with images
translate_pdf(input_pdf_path, output_pdf_path, target_language='sk')

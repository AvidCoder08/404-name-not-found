
import pypdf

def extract_text_from_pdf(pdf_path):
    try:
        reader = pypdf.PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    pdf_path = "/home/riggid/Downloads/PS2_MoneyMuling_GraphTheory - Google Docs.pdf"
    print(extract_text_from_pdf(pdf_path))

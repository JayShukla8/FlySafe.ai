import os
from dotenv import load_dotenv
from fpdf import FPDF
import markdown
from bs4 import BeautifulSoup

def convert_2_pdf(text: str) -> str:
    try:
        # Create output directory if it doesn't exist
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Use a fixed filename
        output_pdf = "output/invoice.pdf"
        
        # Convert Markdown to HTML
        html_content = markdown.markdown(text)

        # Strip HTML tags using BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        plain_text = soup.get_text()

        # Create a PDF object
        pdf = FPDF()
        pdf.add_page()

        # Set font: Arial, regular, size 12
        pdf.set_font("Arial", size=12)

        # Add plain text to the PDF line by line
        lines = plain_text.split('\n')
        for line in lines:
            pdf.cell(200, 10, line, ln=1, align='L')

        # Output the PDF to the specified file
        pdf.output(output_pdf)
        print(f"PDF generated successfully at: {output_pdf}")
        
        # Verify the file was created
        if not os.path.exists(output_pdf):
            raise Exception(f"PDF file was not created at {output_pdf}")
            
        return output_pdf
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        raise e


# convert_2_pdf("Hi How are you doing")
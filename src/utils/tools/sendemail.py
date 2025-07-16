import os
from dotenv import load_dotenv
import sendgrid
import base64
from sendgrid.helpers.mail import Mail, Email, To, Content, Attachment, FileContent, FileName, FileType, Disposition
from typing import Type

from fpdf import FPDF
import markdown
from bs4 import BeautifulSoup

load_dotenv()

# initialize SendGrid
api_key = os.environ.get("SENDGRID_API_KEY")
if not api_key:
    raise ValueError("SENDGRID_API_KEY not found in environment variables")

sg = sendgrid.SendGridAPIClient(api_key=api_key)

class SendEmail:
    def __init__(self, message_content: str, to_email: str):
        self.message_content = message_content
        self.to_email = to_email
        
    def _run(self) -> str:
        try:
            from_email = Email("xyz@gmail.com")
            to = To(self.to_email)
            subject = "Flight Inspection Purchase Invoice"
            content = Content("text/html", self.message_content)

            mail = Mail(from_email, to, subject, content)

            # Attach invoice PDF file
            pdf_path = "output/invoice.pdf"  # Path to the generated invoice PDF
            if not os.path.exists(pdf_path):
                raise Exception(f"Invoice PDF not found at {pdf_path}")

            with open(pdf_path, "rb") as f:
                data = f.read()

            encoded_file = base64.b64encode(data).decode()

            attached_file = Attachment(
                FileContent(encoded_file),
                FileName("invoice.pdf"),
                FileType("application/pdf"),
                Disposition("attachment")
            )

            mail.attachment = attached_file

            # mail_json = mail.get()

            # response = sg.client.mail.send.post(request_body=mail_json)
            response = sg.send(mail)
            print(f"Email status code: {response.status_code}")

            if response.status_code == 202:
                return "Email sent successfully"
            else:
                return f"Error sending email: {response.status_code}"

        except Exception as e:
            return f"Error sending email: {str(e)}"

# Usage example:
# send_email = SendEmail(message_content="Hello", to_email="troy.shubham.tcs@gmail.com")
# result = send_email._run()
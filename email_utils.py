import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import logging
import random
import string

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def generate_otp():
    """Generate a 6-digit OTP"""
    return ''.join(random.choices(string.digits, k=6))

def send_reset_email(to_email: str, otp: str):
    # Get email configuration from environment variables
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    from_email = os.getenv("FROM_EMAIL")

    logger.info(f"Attempting to send email to {to_email}")
    logger.info(f"Using SMTP server: {smtp_server}:{smtp_port}")
    logger.info(f"Using username: {smtp_username}")

    # Create the email message
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = "Password Reset OTP"

    # Create the HTML body with the OTP
    body = f"""
    <html>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
            <div style="background-color: #f8f9fa; border-radius: 5px; padding: 20px; margin-bottom: 20px;">
                <h2 style="color: #007bff; margin-bottom: 20px;">Password Reset Request</h2>
                <p>You have requested to reset your password. Use the following OTP code to reset your password:</p>
                
                <div style="text-align: center; margin: 30px 0;">
                    <div style="background-color: #e9ecef; padding: 20px; border-radius: 5px; font-size: 24px; font-family: monospace; letter-spacing: 5px;">
                        {otp}
                    </div>
                </div>

                <div style="margin-top: 30px; font-size: 0.9em; color: #6c757d;">
                    <p>⚠️ Important Notes:</p>
                    <ul>
                        <li>This OTP will expire in 10 minutes</li>
                        <li>If you didn't request this reset, please ignore this email</li>
                        <li>For security, never share this OTP with anyone</li>
                        <li>Enter this OTP on the password reset page to create a new password</li>
                    </ul>
                </div>

                <p style="margin-top: 30px;">
                    Best regards,<br>
                    Your App Team
                </p>
            </div>
        </body>
    </html>
    """

    msg.attach(MIMEText(body, "html"))

    try:
        logger.info("Attempting to establish SMTP connection...")
        # Create SMTP session
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        
        logger.info("Attempting SMTP login...")
        server.login(smtp_username, smtp_password)
        
        logger.info("Sending email...")
        server.send_message(msg)
        server.quit()
        logger.info("Email sent successfully!")
        return True
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        return False 
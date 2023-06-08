'''
import csv
import smtplib

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# SMTP configuarationsL
SMTP_HOST = "nayana.tk@intel.com"
SMTP_PORT = 587
SMTP_USER =

# Read CSV file and extract dates and email address
csv_file = "data.csv"
today = datetime.today().strftime('%m-%d')
print(today)

birthday_emails = []

with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        # take the index value from header
        email_index = row.index('Email')
        birthday_index = row.index('DOB')
        waniversery_index = row.index('DOJ')
        next(reader) # skip header now

        email = row[email_index]
        birthday = datetime.strptime(row[birthday_index], "%m-%d")
        if birthday.strftime == today:
            birthday_emails.append(email)

if len(birthday_emails) > 0:
    smtp = smtplib.SMTP(SMTP_)
'''
def fun():
    for x in range(22, 23, 24):
        print(x)

fun()
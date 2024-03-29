from urllib.request import urlopen from bs4 import BeautifulSoup from openpyxl import load_workbook 
 
# url = "http://news.bbc.co.uk/2/hi/health/2284783.stm" 
# html = urlopen(url).read() 
 
  
#load excel file 
workbook = load_workbook(filename="20051114.xlsx") 
#open workbook sheet = workbook.active 
  
#modify the desired cell for i in range(1,454):     column='E'+str(i)     html = sheet[column]     print(html)     soup = BeautifulSoup(html.value, features="html.parser") 
 
    # kill all script and style elements     for script in soup(["script", "style"]): 
        script.extract()    # rip it out 
 
    # get text     text = soup.get_text() 
 
    # break into lines and remove leading and trailing space on each     lines = (line.strip() for line in text.splitlines())     # break multi-headlines into a line each     chunks = (phrase.strip() for line in lines for phrase in line.split("  ")) 
    # drop blank lines     text = '\n'.join(chunk for chunk in chunks if chunk)     sheet[column]=text 
    #save the file workbook.save(filename="4.xlsx") 
 
code mbox to csv import mailbox import csv 
 
def mbox_to_csv(mbox_file, csv_file):     with open(csv_file, 'w', newline='', encoding='utf-8') as f:         writer = csv.writer(f)         # Write the header row         writer.writerow(['Subject', 'From', 'To', 'Date', 'Content']) 
 
        with open(mbox_file, 'rb') as mbox: 
            mbox_reader = mailbox.mbox(mbox_file)             for message in mbox_reader: 
                subject = message['subject'] if 'subject' in message else ''                 from_email = message['from'] if 'from' in message else ''                 to_email = message['to'] if 'to' in message else ''                 date = message['date'] if 'date' in message else ''                 if message.is_multipart(): 
                    content = ''.join(str(part.get_payload(decode=True)) for part in message.get_payload())                 else: 
                    content = message.get_payload(decode=True)                 writer.writerow([subject, from_email, to_email, date, content]) 
 
# Convert the .mbox file to .csv file format mbox_file = "enron.mbox" csv_file = "enron.csv" mbox_to_csv(mbox_file, csv_file)

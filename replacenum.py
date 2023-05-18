from num2words import num2words
import re
import pdfrw
from num2words import num2words
import re
import PyPDF2

class Convert:
    def __init__(self) -> None:
        pass
    
    def pdf_to_text(self, filename, output_file):
        with open(filename, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)

            text = ''
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()

        with open(output_file, 'w', encoding='utf-8') as output:
            output.write(text)
            
    def replace_numbers_with_words(self, filename):
        def conv_num(match):
            num = int(match.group())
            return num2words(num)

        # Read the text file
        with open(filename, 'r') as file:
            text = file.read()

        # Replace numbers with words
        replaced_text = re.sub(r'\b\d+\b', conv_num, text)

        # Write the updated text back to the file
        with open(filename, 'w') as file:
            file.write(replaced_text)

        print("Numbers replaced with words in", filename)
        
if __name__ == "__main__":
    convert = Convert()
    convert.pdf_to_text("docs/report1.pdf", "docs/report1.txt")
    convert.replace_numbers_with_words("docs/report1.txt")









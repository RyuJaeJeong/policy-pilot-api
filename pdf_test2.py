from icecream  import ic
from langchain_community.document_loaders import PyPDFium2Loader
import re    

pdf_path = "./static/sample_policy.pdf"    
loader = PyPDFium2Loader(pdf_path)
load = loader.load()

for page in load:    
   # print(f"==================={page.metadata["page"]}===================")
   # print()
   arr = page.page_content.split("\n")
   for text in arr:
      if re.match(r"제\s\d\s장", text): 
         print(f"matched : {text}")
      elif "장" in text:
         print(f"matched : {text}") 
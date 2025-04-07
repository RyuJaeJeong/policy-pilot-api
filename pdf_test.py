from langchain_community.document_loaders import PyPDFLoader
import re

file_path = "./static/sample_policy.pdf"
loader = PyPDFLoader(file_path)
pages = loader.load()

arr = []
for page in pages:
    print(repr(page.page_content))
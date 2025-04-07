import pdfplumber
import json
import re    
    
pdf_path = "./static/sample_policy.pdf"    
pdf = pdfplumber.open(pdf_path)
pages = pdf.pages
arr = []

for page in pages:
    sub = page.extract_text()
    arr.append(sub)

    

jang_text = ""
jang_number = ""
jang_pattern = r"제(\d+)장 (.+)"
jo_number = ""
jo_text = ""
jo_pattern = r"제(\d+)조【(.+)】"

obj = {}
arr3 = [{
    "장":"",
    "장번호":"",
    "조":"",
    "조번호":"",
    "내용":"",
}]

for text in arr:
    arr2 = text.split("\n") 
    for i in range(len(arr2)):
        str = arr2[i]
        if re.match(jang_pattern, str):
            jang_number = re.match(jang_pattern, str).group(1) 
            jang_text = re.match(jang_pattern, str).group(2)
            # print(f"제{jang_number}장 {jang_text}")
        elif re.match(jo_pattern, str): 
            jo_number = re.match(jo_pattern, str).group(1) 
            jo_text = re.match(jo_pattern, str).group(2)
            # print(f"   제{jo_number}조 {jo_text}")
        elif re.match(r"-\s*\d+\s*", str):
            pass    
        else:
            if arr3[-1]["장"] == jang_number and arr3[-1]["조"] == jo_number:
                arr3[-1]["내용"] += str
                pass
            else:
                obj["장"] = jang_number
                obj["장제목"] = jang_text
                obj["조"] = jo_number
                obj["조제목"] = jo_text
                obj["내용"] = str
                arr3.append(obj)
                obj = {}
                pass    
            

print(arr3)

with open("output.json", "w", encoding="utf-8") as f:
    json.dump(arr3, f, ensure_ascii=False, indent=4)
    
    
    
    
         
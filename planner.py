import json
import pandas as pd
import streamlit as st
from azure.identity import DefaultAzureCredential
from langchain_openai import AzureChatOpenAI
import re

st.title("Exception Summary Generator")

# Upload CSV or Excel
uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file with exception records to generate insights using Azure GPT.",
    type=["csv", "xls", "xlsx"],
)

# ---- Utility Functions ---- #

def read_txt_file(file_path):
    """Reads a text file and returns its content as a string."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_method_by_line_number(code_lines, target_line_number):
    method_start = None
    method_end = None
    brace_count = 0
    inside_method = False
    method_name = None
    total_lines = len(code_lines)

    method_pattern = re.compile(
        r'(public|private|protected|internal)?\s+[\w<>\[\],\s]+\s+(?P<name>\w+)\s*\(.*?\)\s*{?'
    )

    st.markdown(f"📄 **Total lines in file**: `{total_lines}`")
    st.markdown(f"🔎 **Searching for method around line**: `{target_line_number}`")

    for i, line in enumerate(code_lines, start=1):
        stripped = line.strip()
        match = method_pattern.match(stripped)

        if match:
            method_start = i
            brace_count = 0
            inside_method = True
            method_name = match.group("name")
            st.markdown(f"🧠 **Method candidate found at line {i}**: `{method_name}`")

        if inside_method:
            brace_count += stripped.count('{') - stripped.count('}')
            if method_start and method_start <= target_line_number <= i:
                for j in range(i, len(code_lines)):
                    brace_count += code_lines[j].count('{') - code_lines[j].count('}')
                    if brace_count == 0:
                        method_end = j + 1
                        method_content = code_lines[method_start - 1:method_end]

                        st.markdown(f"✅ **Method Name**: `{method_name}`")
                        st.markdown(f"📌 **Start Line**: `{method_start}`")
                        st.markdown(f"📌 **End Line**: `{method_end}`")
                        st.markdown("---")
                        return "\n".join(method_content)

    st.warning("⚠️ Could not find a method at or around the given line number.")
    return ""


def format_gpt_output(response):
    """Formats GPT output into markdown-friendly sections."""
    sections = re.split(r"###\s+", response)
    formatted_sections = []
    for section in sections:
        if not section.strip():
            continue
        lines = section.strip().split("\n")
        title = lines[0].strip()
        content = "\n".join(lines[1:]).strip()
        content = re.sub(r"^\s*-\s+", "- ", content, flags=re.MULTILINE)
        content = re.sub(r"^\s*\d+\.\s+", "- ", content, flags=re.MULTILINE)
        formatted_sections.append((title, content))
    return formatted_sections


# ---- Main Summary Generation ---- #

def generate_summary(df):
    credential = DefaultAzureCredential(logging_enable=True)

    azure_gpt = AzureChatOpenAI(
        azure_endpoint="https://aimlameuse2npdopenai.openai.azure.com/",
        openai_api_version="2023-03-15-preview",
        deployment_name="gpt-4o-mini-v2024-07-18-ptu",
        azure_ad_token_provider=lambda: credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        ).token,
        temperature=0.1,
        seed=42,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.1
    )

    # Pick first row for demo
    data_json = df.loc[8].to_json(orient="records")
    code_file_name = df.loc[8].CodeFile
    error_line_num = json.loads(df.loc[8].StackTrace)[0]['line']

    st.markdown(f"🗂️ **Code File:** `{code_file_name}`")
    st.markdown(f"🪵 **Error Line Number:** `{error_line_num}`")

    prompt = f"""
You are an expert software engineer analyzing an exception record from a production environment. Your job is to reason through the failure, provide actionable resolution steps, and suggest a direct fix if relevant code is available.

---
### Root Cause Analysis
Based on the stack trace and exception data provided below, explain in detail why this exception is occurring. Include reasoning based on stack trace level[0], error messages, and any environment clues.

---
### Resolution Steps
List 2–3 potential resolutions to fix this issue. Include short justifications for each option.

---
### Exception Record
{data_json}
"""

    if not pd.isna(code_file_name):
        raw_code = read_txt_file(code_file_name)
        code_lines = raw_code.splitlines()
        extracted_code = extract_method_by_line_number(code_lines, error_line_num)

        prompt += f"""
---
### Relevant Code File (`{code_file_name}`)

Below is the code from the file mentioned in the stack trace. Line numbers are included for precise reference.

[[code]]
{extracted_code}
[[/code]]

---
### Code Fix Suggestion

Based on the above code, the file name and line number from `stacktrace[0]`, suggest a specific **code fix**.

Format your answer as:
- **Affected Line Number**: `Line X`
- **Problem**: Briefly describe the issue.
- **Suggested Fix**: Show the corrected version of that line or a small code block (max 5 lines) if necessary.
- **Reasoning**: Explain *why* this fix works.
"""

    # Call Azure GPT
    response = azure_gpt.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)
    formatted_sections = format_gpt_output(response_text)

    # Display output
    for title, content in formatted_sections:
        st.markdown(f"### {title}")
        st.markdown(content)
        st.markdown("---")


# ---- UI Trigger ---- #

if st.button("Generate Summary"):
    if uploaded_file is not None:
        file_name = uploaded_file.name
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.success("✅ CSV file uploaded successfully!")
        elif file_name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
            st.success("✅ Excel file uploaded successfully!")
        else:
            st.error("❌ Unsupported file type.")
            df = None

        if "df" in locals() and df is not None:
            generate_summary(df)
    else:
        st.warning("⚠️ Please upload a file before submitting.")

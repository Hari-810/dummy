import json
import pandas as pd
import streamlit as st
from azure.identity import DefaultAzureCredential
from langchain_openai import AzureChatOpenAI
import re

st.title("Exception Summary Generator")

# Upload excel file
uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file containing exception records to generate insights using Azure GPT.",
    type=["csv", "xls", "xlsx"],
)

def read_txt_file(file_path):
    """ Reads a text file and returns its content """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()
    

def add_line_numbers(code: str, start: int = 1) -> str:

    lines = code.splitlines()
    numbered_lines = [f"{str(i + start).rjust(4)} | {line}" for i, line in enumerate(lines)]
    return "\n".join(numbered_lines)


def extract_method_by_line_number(code_lines, target_line_number):
    print(code_lines)
    method_start = None
    method_end = None
    brace_count = 0
    inside_method = False
    method_name = None
    total_lines = len(code_lines)

    # Improved method signature pattern with method name capture
    method_pattern = re.compile(
        r'(public|private|protected|internal)?\s+[\w<>\[\],\s]+\s+(?P<name>\w+)\s*\(.*?\)\s*{?'
    )

    for i, line in enumerate(code_lines, start=1):
        stripped = line.strip()

        match = method_pattern.match(stripped)
        if match:
            method_start = i
            brace_count = 0
            inside_method = True
            method_name = match.group("name")

        if inside_method:
            brace_count += stripped.count('{') - stripped.count('}')

            # Check if the target line is within this method's scope
            if method_start and method_start <= target_line_number <= i:
                # Read ahead until method closes
                for j in range(i, len(code_lines)):
                    brace_count += code_lines[j].count('{') - code_lines[j].count('}')
                    if brace_count == 0:
                        method_end = j + 1
                        method_content = code_lines[method_start - 1:method_end]
                        st.markdown("method_name",method_name)
                        print(f"\n✅ Method Name:           {method_name}")
                        print(f"✅ Method starts at line: {method_start}")
                        print(f"✅ Method ends at line:   {method_end}")
                        print(f"📄 Total lines in file:   {total_lines}\n")
                        print("🔍 Extracted Method Content:\n" + "-"*40)
                        # print("".join(method_content))
                        print("-"*40)
                        return "".join(method_content)
                    
def generate_summary(df):
    # Azure credential
    credential = DefaultAzureCredential(logging_enable=True)
    # Initialize Azure GPT
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
    data_json = df.loc[8].to_json(orient="records")
    
    
    code_file_name = df.loc[8].CodeFile
    st.markdown(code_file_name)
    error_line_num = json.loads(df.loc[8].StackTrace)[0]['line']
    st.markdown(error_line_num)

    
    prompt = f"""
    You are an expert software engineer analyzing an exception record from a production environment. Your job is to reason through the failure, provide actionable resolution steps, and suggest a direct fix if relevant code is available.

    Please structure your response as follows:

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
        # numbered_code = add_line_numbers(read_txt_file(code_file_name))  # This is a utility you should implement
        extract_method = extract_method_by_line_number(read_txt_file(code_file_name), error_line_num)
        prompt += f"""

    ---

    ###  Relevant Code File (`{code_file_name}`)

    Below is the code from the file mentioned in the stack trace. Line numbers are included for precise reference.

    [[code]]
    {extract_method}
    [[/code]]

    ---

    ### Code Fix Suggestion

    Based on the above code, the file name and line number from `stacktrace[0]`, suggest a specific **code fix**. 

    Format your answer as:

    - **Affected Line Number**: `Line X`
    - **Problem**: Briefly describe the issue.
    - **Suggested Fix**: Show the corrected version of that line or a small code block (max 5 lines) if necessary.
    - **Reasoning**: Explain *why* this fix works.

    Avoid patch/diff formats. Show complete lines instead.

    """
    

    response = azure_gpt.invoke(prompt)
    response_text = response.content if hasattr(response, "content") else str(response)
    formatted_sections = format_gpt_output(response_text)

    # Display sections one after the othe
    for title, content in formatted_sections:
        st.markdown(f"### {title}")
        st.markdown(content)
        st.markdown("---")

    # print(f"GPT Insights for {df.loc[8]}\n", response)


def format_gpt_output(response):
    # Remove unnecessary headers and split into sections    
    sections = re.split(r"###\s+", response)
    formatted_sections = []
    for section in sections:
        if not section.strip():
            continue
        lines = section.strip().split("\n")
        title = lines[0].strip()
        content = "\n".join(lines[1:]).strip()
        # Convert numbered or dash lists to markdown bullets
        content = re.sub(r"^\s*-\s+", "- ", content, flags=re.MULTILINE)
        content = re.sub(r"^\s*\d+\.\s+", "- ", content, flags=re.MULTILINE)
        formatted_sections.append((title, content))
    return formatted_sections


# Submit button
if st.button("Generate Summary"):
    if uploaded_file is not None:
        file_name = uploaded_file.name
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            st.success("CSV file uploaded successfully!")
        elif file_name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
            st.success("Excel file uploaded successfully!")
        else:
            st.error("Unsupported file type.")
            df = None

        if "df" in locals() and df is not None:
            generate_summary(df)
    else:
        st.warning("Please upload a file before submitting.")

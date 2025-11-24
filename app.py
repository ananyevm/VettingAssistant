import streamlit as st
import openai
from typing import Optional, Dict, Any
import os
from docx import Document
import PyPDF2
import pandas as pd
import io
import json
from pathlib import Path
import shutil

st.set_page_config(
    page_title="ABS Vetting Assistant",
    page_icon="📊",
    layout="wide"
)

# Clean and create gen_code and json folders on app startup
gen_code_dir = "gen_code"
if os.path.exists(gen_code_dir):
    shutil.rmtree(gen_code_dir)
os.makedirs(gen_code_dir, exist_ok=True)

json_dir = "json"
if os.path.exists(json_dir):
    shutil.rmtree(json_dir)
os.makedirs(json_dir, exist_ok=True)

def get_openai_summary(content: str, output_data_info: Optional[Dict[str, Any]] = None) -> Optional[str]:
    """
    Send content to OpenAI GPT-4o for summarization and information assessment
    """
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an expert data analyst. Your task is to:
1. Summarize the provided document highlighting key characteristics, structure, and potential insights.
2. Assess the presence of specific information categories and rate them as follows:
   - 🟢 GREEN: Information is present and comprehensive
   - 🟡 YELLOW: Information is present but insufficient or unclear
   - 🔴 RED: Information is absent or not provided
3. If output data is provided, compare it with the data description and assess data-description match.

Return your response in the following format:

## Information Assessment
**Data type:** [DESCRIPTIVE/MODEL-BASED] [Brief explanation of whether the uploaded data represents descriptive statistics (e.g., descriptive tables, cross-tabulations, frequency distributions) or model-based results (e.g., regression output, statistical model results)]

**Population used to derive results:** [🟢/🟡/🔴] [Brief explanation]
**Method of analysis:** [🟢/🟡/🔴] [Brief explanation]
**Datasets used:** [🟢/🟡/🔴] [Brief explanation]
**Description of the data:** [🟢/🟡/🔴] [Brief explanation]
**Description of all variables:** [🟢/🟡/🔴] [Brief explanation]

## Data-Description Match Assessment (if output data provided)
**Overall data match:** [🟢/🟡/🔴] [Brief explanation of how well the output data matches the description]
**Column match:** [🟢/🟡/🔴] [Analysis of whether columns in data match variables described in document]
**Data types match:** [🟢/🟡/🔴] [Assessment of whether data types align with expectations]
**Sample size match:** [🟢/🟡/🔴] [Comparison of actual vs expected sample size]"""
                },
                {
                    "role": "user", 
                    "content": f"""Please analyze this document and provide both a summary and assessment of the following required information:

1. Population used to derive results
2. Method of analysis
3. Datasets used
4. Description of the data
5. Description of all the variables

Document content:
{content}

{f'''
Output Data Information (if provided):
- Shape: {output_data_info["shape"]} (rows, columns)
- Columns: {output_data_info["columns"]}
- Data types: {output_data_info["dtypes"]}
- Null counts: {output_data_info["null_counts"]}
- Sample data (first 5 rows): {output_data_info["sample_data"]}
- Summary statistics: {output_data_info["summary_stats"]}

Please compare this actual output data with the data description and assess how well they match.
''' if output_data_info and "error" not in output_data_info else ""}"""
                }
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error connecting to OpenAI: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file) -> str:
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def extract_text_from_docx(docx_file) -> str:
    """Extract text from DOCX file"""
    try:
        doc = Document(io.BytesIO(docx_file.read()))
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
        return ""

def clean_table_with_openai(raw_data: str, filename: str) -> Optional[str]:
    """Use OpenAI to clean and structure table data"""
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        prompt = f"""
You are a data cleaning expert. I have raw table data extracted from an Excel file that may contain:
- Merged cells
- Headers in wrong positions
- Empty rows/columns
- Inconsistent formatting
- Extraneous information

Please analyze this raw data and extract clean, well-structured tabular data in CSV format.

Raw data from file '{filename}':
{raw_data}

Requirements:
1. Identify the actual table structure and headers
2. Remove any extraneous information
3. Ensure consistent data types in each column
4. Handle any merged cells or formatting issues
5. Return ONLY the clean CSV data (no explanations)
6. Use proper column headers
7. Remove any completely empty rows or columns

Return the cleaned data as a proper CSV format.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data cleaning expert. Return only clean CSV data with no additional text or explanations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        cleaned_data = response.choices[0].message.content.strip()
        
        # Remove markdown code block markers if present
        if cleaned_data.startswith('```csv'):
            cleaned_data = cleaned_data[6:]
        if cleaned_data.startswith('```'):
            cleaned_data = cleaned_data[3:]
        if cleaned_data.endswith('```'):
            cleaned_data = cleaned_data[:-3]
        
        return cleaned_data.strip()
        
    except Exception as e:
        st.error(f"Error cleaning data with OpenAI: {str(e)}")
        return None

def extract_data_from_csv_xlsx(data_file) -> Dict[str, Any]:
    """Extract data and metadata from CSV/XLSX files with OpenAI cleaning"""
    try:
        # Create intermediate directory
        intermediate_dir = "intermediate"
        os.makedirs(intermediate_dir, exist_ok=True)
        
        # Reset file pointer
        data_file.seek(0)
        
        # Read raw data first for OpenAI cleaning
        if data_file.type == "text/csv":
            # For CSV, read as text first
            content = str(data_file.read(), "utf-8")
            data_file.seek(0)  # Reset for pandas
            df_raw = pd.read_csv(io.BytesIO(data_file.read()))
        elif data_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            # For Excel, read raw data as text representation
            data_file.seek(0)
            df_raw = pd.read_excel(io.BytesIO(data_file.read()), header=None)
            content = df_raw.to_string(index=False, header=False)
        else:
            return {"error": "Unsupported file type"}
        
        # Clean data with OpenAI
        cleaned_csv_content = clean_table_with_openai(content, data_file.name)
        
        if cleaned_csv_content:
            # Save cleaned data to intermediate folder
            cleaned_filename = f"{Path(data_file.name).stem}_cleaned.csv"
            cleaned_filepath = os.path.join(intermediate_dir, cleaned_filename)
            
            with open(cleaned_filepath, 'w', encoding='utf-8') as f:
                f.write(cleaned_csv_content)
            
            # Read cleaned data for analysis
            df = pd.read_csv(io.StringIO(cleaned_csv_content))
            
        else:
            # Fallback to original data if cleaning fails
            if data_file.type == "text/csv":
                data_file.seek(0)
                df = pd.read_csv(io.BytesIO(data_file.read()))
            else:
                data_file.seek(0)
                df = pd.read_excel(io.BytesIO(data_file.read()))
        
        # Get basic info about the dataset
        info = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "sample_data": df.head().to_dict(),
            "null_counts": df.isnull().sum().to_dict(),
            "summary_stats": df.describe().to_dict() if len(df.select_dtypes(include=['number']).columns) > 0 else {},
            "cleaned_file_path": cleaned_filepath if cleaned_csv_content else None
        }
        
        return info
        
    except Exception as e:
        return {"error": f"Error reading data file: {str(e)}"}

def generate_rule_of_10_checker(cleaned_file_path: str, filename: str) -> Optional[str]:
    """Use OpenAI to generate Python code for rule of 10 checking"""
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        prompt = f"""
Create a Python script that checks the "rule of 10" for a CSV file. The rule of 10 states that all cells in the table should either be empty or contain a number that is 10 or larger.

Requirements:
1. Read the CSV file from: {cleaned_file_path}
2. Check every cell in the table (excluding headers)
3. For each violation, record:
   - Row number
   - Column name
   - Cell value (convert to Python native types for JSON serialization)
   - Cell position (row, col)
4. Save results to a JSON file called "json/rule_of_10_violations.json"
5. The JSON should contain a list of violations with the structure:
   {{"violations": [{{"row": int, "column": str, "value": any, "position": [int, int]}}]}}

The script should:
- Handle different data types (numbers, strings, empty cells)
- Consider empty/NaN cells as compliant (PASS)
- Consider cells with numbers >= 10 as compliant (PASS)
- Only flag cells with numbers < 10 as violations (FAIL)
- Ignore non-numeric values (strings are neither pass nor fail)
- Handle numpy data types (numpy.int64, numpy.float64, etc.) properly
- Convert pandas/numpy data types to Python native types before JSON serialization
- Create the json directory if it doesn't exist
- Print summary statistics
- Return the path to the JSON results file

Generate a complete Python script named "rule_of_10_checker.py".
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Python expert. Generate ONLY valid Python code with no explanations, comments, or markdown. Return raw Python code that can be executed directly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        code_content = response.choices[0].message.content.strip()
        
        # Remove markdown code block markers if present
        if code_content.startswith('```python'):
            code_content = code_content[9:]
        if code_content.startswith('```'):
            code_content = code_content[3:]
        if code_content.endswith('```'):
            code_content = code_content[:-3]
        
        return code_content.strip()
        
    except Exception as e:
        st.error(f"Error generating rule of 10 checker: {str(e)}")
        return None

def generate_dominance_checker(cleaned_file_path: str, filename: str) -> Optional[str]:
    """Use OpenAI to generate Python code for dominance rule checking"""
    try:
        client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        prompt = f"""
Create a Python script that checks the "dominance rule" for a CSV file. The dominance rule checks if any cell value divided by its row sum exceeds 0.8 (80%).

Requirements:
1. Read the CSV file from: {cleaned_file_path}
2. For each row, calculate the sum of all numeric values
3. For each cell in that row, check if (cell_value / row_sum) > 0.8
4. For each violation, record:
   - Row number
   - Column name
   - Cell value (convert to Python native types for JSON serialization)
   - Row sum
   - Dominance ratio (cell_value / row_sum)
   - Cell position (row, col)
5. Save results to a JSON file called "json/dominance_violations.json"
6. The JSON should contain a list of violations with the structure:
   {{"violations": [{{"row": int, "column": str, "value": any, "row_sum": float, "dominance_ratio": float, "position": [int, int]}}]}}

The script should:
- Handle different data types (numbers, strings, empty cells)
- Skip rows where the sum is 0 or NaN (cannot calculate dominance)
- Only consider numeric values for calculations
- Ignore non-numeric values in dominance calculations
- IMPORTANT: When iterating over rows with df.iterrows(), remember that 'row' is a pandas Series, NOT a DataFrame
- NEVER use row.select_dtypes() - this method doesn't exist on Series objects
- Instead, use list comprehension or isinstance() to filter numeric values from the row Series
- Handle numpy data types (numpy.int64, numpy.float64, etc.) properly
- Convert pandas/numpy data types to Python native types before JSON serialization
- Create the json directory if it doesn't exist
- Print summary statistics including total rows checked and violations found
- Return the path to the JSON results file

Generate a complete Python script named "dominance_checker.py".
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Python expert. Generate ONLY valid Python code with no explanations, comments, or markdown. Return raw Python code that can be executed directly."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        code_content = response.choices[0].message.content.strip()
        
        # Remove markdown code block markers if present
        if code_content.startswith('```python'):
            code_content = code_content[9:]
        if code_content.startswith('```'):
            code_content = code_content[3:]
        if code_content.endswith('```'):
            code_content = code_content[:-3]
        
        return code_content.strip()
        
    except Exception as e:
        st.error(f"Error generating dominance checker: {str(e)}")
        return None

def process_supporting_data(supporting_file) -> Dict[str, Any]:
    """Process supporting data file without running rule of 10 check"""
    try:
        # Extract and clean the supporting data
        supporting_info = extract_data_from_csv_xlsx(supporting_file)
        
        if "error" in supporting_info:
            return {"error": supporting_info["error"]}
        
        cleaned_file_path = supporting_info.get("cleaned_file_path")
        if not cleaned_file_path:
            return {"error": "Failed to get cleaned file path"}
        
        return {
            "success": True,
            "supporting_info": supporting_info,
            "cleaned_file_path": cleaned_file_path
        }
            
    except Exception as e:
        return {"error": f"Error processing supporting data: {str(e)}"}

def run_rule_of_10_check(cleaned_file_path: str, filename: str) -> Dict[str, Any]:
    """Generate and run rule of 10 checker"""
    try:
        # Generate rule of 10 checker
        checker_code = generate_rule_of_10_checker(cleaned_file_path, filename)
        
        if not checker_code:
            return {"error": "Failed to generate checker code"}
        
        # Save the checker script in gen_code folder
        checker_path = os.path.join("gen_code", "rule_of_10_checker.py")
        with open(checker_path, 'w', encoding='utf-8') as f:
            f.write(checker_code)
        
        # Execute the checker
        result = os.system(f"python3 {checker_path}")
        
        if result == 0:
            # Read the results JSON
            results_path = "json/rule_of_10_violations.json"
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    violations_data = json.loads(f.read())
                return {
                    "success": True,
                    "violations": violations_data,
                    "checker_path": checker_path,
                    "results_path": results_path
                }
            else:
                return {"error": "Results file not found"}
        else:
            return {"error": "Rule of 10 checker execution failed"}
            
    except Exception as e:
        return {"error": f"Error running rule of 10 check: {str(e)}"}

def run_dominance_check(cleaned_file_path: str, filename: str) -> Dict[str, Any]:
    """Generate and run dominance rule checker"""
    try:
        # Generate dominance checker
        checker_code = generate_dominance_checker(cleaned_file_path, filename)
        
        if not checker_code:
            return {"error": "Failed to generate dominance checker code"}
        
        # Save the checker script in gen_code folder
        checker_path = os.path.join("gen_code", "dominance_checker.py")
        with open(checker_path, 'w', encoding='utf-8') as f:
            f.write(checker_code)
        
        # Execute the checker
        result = os.system(f"python3 {checker_path}")
        
        if result == 0:
            # Read the results JSON
            results_path = "json/dominance_violations.json"
            if os.path.exists(results_path):
                with open(results_path, 'r') as f:
                    violations_data = json.loads(f.read())
                return {
                    "success": True,
                    "violations": violations_data,
                    "checker_path": checker_path,
                    "results_path": results_path
                }
            else:
                return {"error": "Dominance results file not found"}
        else:
            return {"error": "Dominance checker execution failed"}
            
    except Exception as e:
        return {"error": f"Error running dominance check: {str(e)}"}

def main():
    st.title("ABS Vetting Assistant")
    
    # File upload sections - vertically stacked
    st.subheader("Data Description")
    uploaded_file = st.file_uploader(
        "Upload Data Description",
        type=['txt', 'pdf', 'docx', 'md'],
        help="Upload a file containing your data description"
    )
    
    st.subheader("Output Data")
    output_data_file = st.file_uploader(
        "Upload Output Data",
        type=['csv', 'xlsx'],
        help="Upload CSV or XLSX file containing the actual output data"
    )
    
    st.subheader("Supporting Data")
    supporting_data_file = st.file_uploader(
        "Upload Supporting Data",
        type=['csv', 'xlsx'],
        help="Upload CSV or XLSX file containing supporting data for rule of 10 check"
    )
    
    if uploaded_file is not None:
        # Read file content
        try:
            if uploaded_file.type == "text/plain":
                content = str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "application/pdf":
                content = extract_text_from_pdf(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                content = extract_text_from_docx(uploaded_file)
            else:
                content = str(uploaded_file.read(), "utf-8")
            
            
            # Extract output data info if provided
            output_data_info = None
            if output_data_file is not None:
                output_data_info = extract_data_from_csv_xlsx(output_data_file)
                
                if "error" in output_data_info:
                    st.error(output_data_info["error"])
                    output_data_info = None

            # Process supporting data if provided
            supporting_data_results = None
            if supporting_data_file is not None:
                supporting_data_results = process_supporting_data(supporting_data_file)
                
                if "error" in supporting_data_results:
                    st.error(supporting_data_results["error"])
                    supporting_data_results = None

            # Check research output button
            if st.button("Check Research Output", type="primary"):
                with st.spinner("Analyzing data description and comparing with output data..."):
                    summary = get_openai_summary(content, output_data_info)
                    
                    if summary:
                        st.markdown(summary)
                    else:
                        st.error("Failed to generate analysis. Please check your OpenAI API key and try again.")
                
                # Run rule of 10 check if supporting data is available
                if supporting_data_results is not None:
                    st.markdown("## 🔍 Rule of 10 Check")
                    rule_of_10_results = run_rule_of_10_check(
                        supporting_data_results["cleaned_file_path"], 
                        supporting_data_file.name
                    )
                    
                    if "error" in rule_of_10_results:
                        st.error(rule_of_10_results["error"])
                    else:
                        # Display rule of 10 results
                        violations = rule_of_10_results["violations"]
                        violation_count = len(violations["violations"])
                        
                        if violation_count == 0:
                            st.success("🟢 **Rule of 10 Check: PASSED** - No violations found!")
                        else:
                            st.error(f"🔴 **Rule of 10 Check: FAILED** - Found {violation_count} violations")
                            
                            # Show violations in expander
                            with st.expander(f"View Rule of 10 Violations ({violation_count})"):
                                violations_df = pd.DataFrame(violations["violations"])
                                st.dataframe(violations_df)
                                
                                # Show summary stats
                                st.write("**Violation Summary:**")
                                if not violations_df.empty:
                                    col_violations = violations_df['column'].value_counts()
                                    st.write("Violations by column:")
                                    for col, count in col_violations.items():
                                        st.write(f"- {col}: {count} violations")
                    
                    # Run dominance check after rule of 10
                    st.markdown("## 🚨 Dominance Rule Check")
                    dominance_results = run_dominance_check(
                        supporting_data_results["cleaned_file_path"], 
                        supporting_data_file.name
                    )
                    
                    if "error" in dominance_results:
                        st.error(dominance_results["error"])
                    else:
                        # Display dominance results
                        dom_violations = dominance_results["violations"]
                        dom_violation_count = len(dom_violations["violations"])
                        
                        if dom_violation_count == 0:
                            st.success("🟢 **Dominance Rule Check: PASSED** - No violations found!")
                        else:
                            st.error(f"🔴 **DOMINANCE RULE VIOLATION - RED ALERT!** - Found {dom_violation_count} violations where cell values exceed 80% of row sum")
                            
                            # Show dominance violations in expander
                            with st.expander(f"View Dominance Rule Violations ({dom_violation_count})"):
                                dom_violations_df = pd.DataFrame(dom_violations["violations"])
                                st.dataframe(dom_violations_df)
                                
                                # Show dominance summary stats
                                st.write("**Dominance Violation Summary:**")
                                if not dom_violations_df.empty:
                                    col_dom_violations = dom_violations_df['column'].value_counts()
                                    st.write("Violations by column:")
                                    for col, count in col_dom_violations.items():
                                        st.write(f"- {col}: {count} violations")
                                    
                                    # Show max dominance ratios
                                    if 'dominance_ratio' in dom_violations_df.columns:
                                        max_ratio = dom_violations_df['dominance_ratio'].max()
                                        st.write(f"**Highest dominance ratio:** {max_ratio:.3f} ({max_ratio*100:.1f}%)")
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    

if __name__ == "__main__":
    main()
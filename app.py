import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Optional, List
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ---------------------------
# LLM Setup
# ---------------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5
)

# ---------------------------
# Pydantic Schema
# ---------------------------
class Information(BaseModel):
    name: str = Field(description="Name of the candidate.")
    studies: List[str] = Field(description="Qualification of the candidate.")
    experience: Optional[List[str]] = Field(description="Companies where the candidate worked.")
    skills: Optional[List[str]] = Field(description="Skills of the candidate.")
    total_work_experience: int = Field(description="Total years of work experience.")

parser = PydanticOutputParser(pydantic_object=Information)

# ---------------------------
# Prompt Template
# ---------------------------
template = PromptTemplate(
    template="""
Extract the required structured information from the following resume.

{introduction}

{format_instruction}
""",
    input_variables=["introduction"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Resume Information Extractor", page_icon="📄")

st.title("📄 Resume Information Extractor")
st.write("Paste a resume introduction below to extract structured information.")

resume_text = st.text_area("Paste Resume Here", height=400)

if st.button("Extract Information"):
    if resume_text.strip() == "":
        st.warning("Please paste the resume content first.")
    else:
        with st.spinner("Processing..."):
            try:
                # Create prompt
                prompt = template.invoke({"introduction": resume_text})

                # Call LLM
                result = llm.invoke(prompt)

                # Parse output
                structured_output = parser.parse(result.content)

                # ---------------------------
                # Display Results
                # ---------------------------
                st.success("Information Extracted Successfully!")

                st.subheader("📌 Extracted Details")

                st.write("**Name:**", structured_output.name)
                st.write("**Qualifications:**", structured_output.studies)
                st.write("**Experience:**", structured_output.experience)
                st.write("**Skills:**", structured_output.skills)
                st.write("**Total Work Experience (Years):**", structured_output.total_work_experience)

            except Exception as e:
                st.error(f"Error: {e}")
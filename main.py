from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from typing import TypedDict, Optional,Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm=ChatGroq(model="openai/gpt-oss-120b",temperature=0.5)

class information(BaseModel):
    name:str=Field(default=None,description="Name of the candidate.")
    studies:list[str]=Field(default=None,description="Qualification of the candidate.")
    experience:Optional[list[str]]=Field(description="Companies where the candidate work.")
    skills:Optional[list[str]]=Field(description="Skills of the candidate which are given.")
    total_work_experience:int=Field(description="how many years the candidate works.")

parser=PydanticOutputParser(pydantic_object=information)

template=PromptTemplate(
    template="""{introduction} \n {format_instruction}""",
    input_variables=['introduction'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

intro="""AARAV SHARMA

        📍 New Delhi, India
        📞 +91-98765-43210
        📧 aarav.sharma@email.com

        🔗 LinkedIn: linkedin.com/in/aaravsharma
        🔗 GitHub: github.com/aaravsharma

        PROFESSIONAL SUMMARY

        Detail-oriented and analytical Data Science graduate with strong foundations in mathematics, statistics, and machine learning. Skilled in Python, SQL, and data visualization tools. Passionate about solving real-world business problems using data-driven approaches. Seeking an entry-level Data Analyst / Data Scientist role to contribute analytical skills and innovative thinking to organizational growth.

        EDUCATION

        Bachelor of Technology (B.Tech) – Computer Science
        Delhi Technological University, Delhi
        2021 – 2025
        CGPA: 8.7/10

        Relevant Coursework:
        Data Structures & Algorithms, Machine Learning, Probability & Statistics, Database Management Systems, Artificial Intelligence.

        TECHNICAL SKILLS

        Programming: Python, SQL, R

        Libraries & Frameworks: Pandas, NumPy, Scikit-learn, TensorFlow

        Data Visualization: Power BI, Tableau, Matplotlib

        Tools: Git, Jupyter Notebook, Excel

        Databases: MySQL, PostgreSQL

        PROJECTS
        🔹 Sales Forecasting Model

        Built a time-series forecasting model using ARIMA and Prophet.

        Improved demand prediction accuracy by 18%.

        Used Python and Power BI for data analysis and dashboard reporting.

        🔹 Customer Segmentation using K-Means

        Performed clustering analysis to identify customer segments.

        Helped simulate targeted marketing strategies.

        Achieved 25% improved marketing efficiency in case simulation.

        🔹 Resume Screening using NLP

        Developed a basic NLP-based resume parser using Python.

        Extracted skills and experience automatically from PDF resumes.

        INTERNSHIP EXPERIENCE

        Data Analyst Intern
        ABC Analytics Pvt. Ltd., Gurgaon
        May 2024 – July 2024

        Cleaned and processed large datasets (50k+ records).

        Created interactive dashboards for sales and performance metrics.

        Automated weekly reporting, reducing manual effort by 40%.

        CERTIFICATIONS

        Google Data Analytics Professional Certificate

        Machine Learning by Andrew Ng (Coursera)

        SQL for Data Science (Udemy)

        ACHIEVEMENTS

        Finalist – National Level Hackathon (2024)

        Presented research paper at College Tech Symposium

        Ranked Top 5% in university coding contest

        SOFT SKILLS

        Problem-Solving

        Analytical Thinking

        Team Collaboration

        Communication Skills

        Time Management

        If you want, I can also create:

        A fresher IT resume version

        A finance-focused resume

        A sales/marketing profile

        A 1-page ATS optimized resume format

        Or convert this into a PDF-ready professional layout

        Just tell me the role you want 😊"""

temp=template.invoke({'introduction':intro})

result=llm.invoke(temp)

main_result=parser.parse(result.content)

print("Name:-",main_result.name)
print("Qualifications:-",main_result.studies)
print("Experience:-",main_result.experience)
print("Skills:-",main_result.skills)
print("Year_work_experience:-",main_result.total_work_experience)
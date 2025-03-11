import streamlit as st 
import os
import io
import tempfile
import openai
import re
from dotenv import load_dotenv
import textwrap

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.output_parsers import RegexParser
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyMuPDFLoader 


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("Please set your OpenAI API key in the environment variable 'OPENAI_API_KEY'.")
    st.stop()

client = openai.Client(api_key=openai_api_key)

# function used to generate response from chatgpt
def call_chat_completion(system_message: str, user_message: str, model="gpt-4o", temperature=0.8) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

st.title("Financial Document Helper")
st.write("This is an AI-powered assitant to answer your questions for financial documents.")

uploaded_file = st.file_uploader("Upload the financial document (pdf file only)", type=["pdf"])
question_input = st.text_area(label="Your question", placeholder="Enter the question you want to know from the document.", height=150)


def getanswer(query):
    relevant_chunks = docembeddings.similarity_search_with_score(query,k=2)
    chunk_docs=[]
    for chunk in relevant_chunks:
        chunk_docs.append(chunk[0])
    results = chain({"input_documents": chunk_docs, "question": query})
    text_reference=""
    for i in range(len(results["input_documents"])):
        text_reference+=results["input_documents"][i].page_content
    output={"Answer":results["output_text"],"Reference":text_reference}
    return output

tone = st.selectbox(
    "Choose the tone of the answer",
    ['Casual', 'Professional', "Concise", "Elaborated"],
    index = 0
)

def escape_markdown(text):
    """Escape Markdown special characters in LLM output"""
    return re.sub(r"([_*])", r"\\\1", text)

if st.button("Get the Answer"):
    if uploaded_file is None:
        st.error("Please upload a PDF file first for your question")
    else:
        if question_input is None or question_input.strip() == "":
            st.error("Please enter your question for the document")
        else:
            st.info("Please wait, Processing the answer...")
            # temperoraly save the uploaded file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name  # Store file path

            # load the uploaded file
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # split the uploaded file
            chunk_size_value = 1000
            chunk_overlap = 100
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size_value, chunk_overlap=chunk_overlap, length_function=len
            )
            texts = text_splitter.split_documents(documents)

            # embed the uploaded file and store it to the vector database
            docembeddings = FAISS.from_documents(texts, OpenAIEmbeddings())
            docembeddings.save_local("llm_faiss_index")

            docembeddings = FAISS.load_local(
                "llm_faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
            
            prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
                If the context is not finance-related, reply that the context is not finance-related and is out of the scope of the assistant's responsibility. 

                This must be in the following format:

                Question: [question here]
                Helpful Answer: [answer here]
                Score: [score between 0 and 100]

                Make sure to ALWAYS include a numerical Score value after the Answer in the required format.

                Begin!

                Context:
                ---------
                {context}
                ---------
                Question: {question}
                Helpful Answer:"""
            
            output_parser = RegexParser(
                regex=r"([\s\S]+?)\s*Score:\s*([\d]+)",
                output_keys=["answer", "score"],
            )

            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"],
                output_parser=output_parser
            )
            chain = load_qa_chain(OpenAI(temperature=0), chain_type="map_rerank", return_intermediate_steps=True, prompt=PROMPT)

            # get the answer for the question from the vector database
            result = getanswer(question_input)
            st.write("Here is the answer to your question: ")

            # further process the answer to the user
            prompt = f"The question to answer is {question_input}, and the fact for the answer we get from the financial document is {result["Answer"]}, generate a reply in {tone} tone to the user in plain text format without any markdown simbols"
            sys_message = "you are a finance professional who help people understanding financial documents and answer their questions"
            response = call_chat_completion(sys_message, prompt)

            wrapped_text = textwrap.fill(response, width=80) 
            st.code(wrapped_text, language="text")


            

    



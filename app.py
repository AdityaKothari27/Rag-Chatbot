import os
import streamlit as st
# from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmltemplates import css, bot_template, user_template
from langchain_huggingface.llms import HuggingFaceEndpoint
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage



listt =[]


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        #Limiting to the first 5 pages for testing
        # num_pages = min(len(pdf_reader.pages), 5)
        num_pages = len(pdf_reader.pages)
        for page in pdf_reader.pages[:num_pages]:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=20000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    try:

        st.write("Loading HuggingFace embeddings model...")
        # embeddings = SentenceTransformer("intfloat/e5-mistral-7b-instruct")
        # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        embeddings = HuggingFaceInstructEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # embeddings = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True)
        st.write("Creating vector store with FAISS...")
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        st.write("Vector store created successfully.")
        return vectorstore
    except Exception as e:
        st.error(f"Error in creating vector store: {e}")
        return None
    

def get_llm(api_token):
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_token)


def get_conversation_chain(vectorstore, api_token):
    try:
        st.write("Loading LLM...")
        # llm = HuggingFaceEndpoint(
        #     # repo_id = "meta-llama/Meta-Llama-3-8B",
        #     # repo_id = "google/flan-t5-base",
        #     # repo_id = "facebook/bart-large-cnn",
        #     # repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        #     # repo_id = "mistralai/Mistral-7B-Instruct-v0.3",
        #     # repo_id = "openai-community/gpt2", small context size
        #     # repo_id= "facebook/bart-large-xsum", kwargs error
        #     # repo_id = "google/gemma-2-9b", model too large (36gb/10gb)
        #     # repo_id = "Qwen/Qwen2-72B-Instruct" model too large(145gb/10gb),
        #     # repo_id="mistralai/Mistral-7B-v0.3", model too large (14GB/10GB)
        #     repo_id = "mistralai/Mistral-7B-Instruct-v0.3",
        #     api_key=api_token,
        #     temperature=0.7,
        #     max_new_tokens=500,
        #     model_kwargs={"max_length": 600}
        # )
        # llm = ChatGroq(groq_api_key=api_token, model_name='mixtral-8x7b-32768', temperature=0.4)
        # llm = ChatGroq(groq_api_key=api_token, model_name='llama3-8b-8192', temperature=0.4)
        # llm = ChatGroq(groq_api_key=api_token, model_name='gemma-7b-it', temperature=0.4)
        # llm = ChatGroq(groq_api_key=api_token, model_name='llama3-70b-8192', temperature=0.4)
        llm = get_llm(api_token)
        st.write("Creating ConversationalRetrievalChain...")
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        st.write("Conversational chain created successfully.")
        st.toast("You can enter the prompt now")
        return conversation_chain
    except Exception as e:
        st.error(f"Error in creating conversation chain: {e}")
        return None


def get_simple_chain(api_token):
    llm = get_llm(api_token)
    prompt = PromptTemplate(
        input_variables=["question"],
        template="{question}"
    )
    return LLMChain(llm=llm, prompt=prompt)


# def handle_userinput(user_question):
#     if st.session_state.conversation is None:
#         st.error("Conversation chain is not initialized.")
#         return

#     try:
#         response = st.session_state.conversation({'question': user_question})
#         st.session_state.chat_history = response['chat_history']

#         for i, message in enumerate(st.session_state.chat_history):
#             if i % 2 == 0:
#                 st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#                 # st.write(message.content)
#             else:
#                 st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
#                 # st.write(message.content)
#     except Exception as e:
#         st.error(f"Error in handling user input: {e}")

def handle_userinput(user_question):
    if 'chat_history' not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []

    if st.session_state.conversation:
        try:
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
        except Exception as e:
            st.error(f"Error in handling user input: {e}")
            return
    else:
        try:
            response = st.session_state.simple_chain.run(user_question)
            st.session_state.chat_history.extend([
                HumanMessage(content=user_question),
                AIMessage(content=response)
            ])
        except Exception as e:
            st.error(f"Error in handling user input: {e}")
            return

    for i, message in enumerate(st.session_state.chat_history):
        if isinstance(message, HumanMessage):
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        elif isinstance(message, AIMessage):
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():

    #environment variables
    # load_dotenv()
    
    st.set_page_config(page_title="Wellness AI", page_icon="🥗")

    # st.markdown(
    # """
    # <style>
    # .reportview-container {
    #     background: url("https://w0.peakpx.com/wallpaper/375/1008/HD-wallpaper-blackness-black-plain-thumbnail.jpg")
    # }
    # .sidebar .sidebar-content {
    #     background: url("https://w0.peakpx.com/wallpaper/375/1008/HD-wallpaper-blackness-black-plain-thumbnail.jpg")
    # }
    # </style>
    # """,
    # unsafe_allow_html=True
    # )
    


    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "simple_chain" not in st.session_state:
        st.session_state.simple_chain = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Wellness AI 🥗")
    
    gemini_api_token = st.secrets["GEMINI_API_KEY"]
    # gemini_api_token = os.getenv("GEMINI_API_KEY")
    if not gemini_api_token:
        st.error("Gemini API token is not set")
    else:
        if not st.session_state.simple_chain:
            st.session_state.simple_chain = get_simple_chain(gemini_api_token)

    user_question = st.chat_input("Ask a question:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Document:")
        pdf_docs = st.file_uploader("Upload your PDF file", type="pdf", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                st.write("Extracted text from PDF:")
                st.write(raw_text[:1000]+"...")
                
                text_chunks = get_text_chunks(raw_text)
                st.write(len(text_chunks))
                # st.write("Text chunks:")
                # st.write(text_chunks)
                
                vectorstore = get_vectorstore(text_chunks)
                if vectorstore:
                    st.session_state.conversation = get_conversation_chain(vectorstore, gemini_api_token)
                    st.session_state.simple_chain = None  # Disable simple chain when using RAG
                else:
                    st.session_state.conversation = None
        
        st.sidebar.markdown("""
                ### Quick Nutrition Tips:
                - Stay hydrated
                - Eat a variety of colorful fruits and vegetables
                - Choose whole grains over refined grains
                - Limit processed foods and added sugars
                """)

if __name__ == '__main__':
    main()




# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Doc Summariser", page_icon="🤖")

#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None

#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Doc Summariser ✨")
#     user_question = st.chat_input("Ask a question about your document(s):")
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Your document:")
#         pdf_docs = st.file_uploader("Upload your PDF file", type="pdf", accept_multiple_files=True)
#         if st.button("Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 st.write("Extracted text from PDF:")
#                 st.write(raw_text[:1000]+"...")
                
#                 text_chunks = get_text_chunks(raw_text)
#                 st.write(len(text_chunks))
#                 # st.write("Text chunks:")
#                 # st.write(text_chunks)

#                 vectorstore = get_vectorstore(text_chunks)
#                 if vectorstore:
#                     hf_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#                     groq_api_key = os.getenv("GROQ_API_KEY")
#                     gemini_api_token = os.getenv("GEMINI_API_KEY")
#                     if not hf_api_token:
#                         st.error("Hugging Face API token is not set.")
#                         return
#                     elif not groq_api_key:
#                         st.error("Groq API token is not set")
#                     elif not gemini_api_token:
#                         st.error("Gemini API token is not set")
#                     else:
#                         st.session_state.conversation = get_conversation_chain(vectorstore, gemini_api_token)
#                 else:
#                     st.session_state.conversation = None

# if __name__ == '__main__':
#     main()
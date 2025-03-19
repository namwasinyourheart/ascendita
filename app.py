from typing import Callable, TypeVar
import os
import inspect
import shutil
import streamlit as st
import streamlit_analytics2 as streamlit_analytics
from dotenv import load_dotenv
from streamlit_chat import message
from streamlit_pills import pills  # (no longer used for query selection)
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
from streamlit.delta_generator import DeltaGenerator
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from custom_callback_handler import CustomStreamlitCallbackHandler
from agents import define_graph




def initialize_callback_handler(main_container: DeltaGenerator):
    V = TypeVar("V")
    def wrap_function(func: Callable[..., V]) -> Callable[..., V]:
        context = get_script_run_ctx()
        def wrapped(*args, **kwargs) -> V:
            add_script_run_ctx(ctx=context)
            return func(*args, **kwargs)
        return wrapped
    streamlit_callback_instance = CustomStreamlitCallbackHandler(parent_container=main_container)
    for method_name, method in inspect.getmembers(streamlit_callback_instance, predicate=inspect.ismethod):
        setattr(streamlit_callback_instance, method_name, wrap_function(method))
    return streamlit_callback_instance

def execute_chat_conversation(user_input, graph, output_container):
    callback_handler_instance = initialize_callback_handler(output_container)
    try:
        output = graph.invoke(
            {
                "messages": list(message_history.messages) + [user_input],
                "user_input": user_input,
                "config": settings,
                "callback": callback_handler_instance,
            },
            {"recursion_limit": 30},
        )
        message_output = output.get("messages")[-1]
        messages_list = output.get("messages")
        message_history.clear()
        message_history.add_messages(messages_list)
    except Exception as exc:
        print(exc)
        return ":( Sorry, some error occurred. Can you please try again?"
    # Convert the output to a string before returning it.
    return message_output.content if hasattr(message_output, "content") else str(message_output)

def main():
    load_dotenv()
    st.set_page_config(layout="wide")

    st.markdown(
        """
        <style>
        /* Adjust font-size for paragraphs inside chat message containers */
        .stChatMessage p {
            font-size: 6px;  /* adjust as needed */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("🚀 Ascendita: Helping you rise in your career")
    streamlit_analytics.start_tracking()

    # # Set environment variables
    # os.environ["LINKEDIN_EMAIL"] = st.secrets.get("LINKEDIN_EMAIL", "")
    # os.environ["LINKEDIN_PASS"] = st.secrets.get("LINKEDIN_PASS", "")
    # os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "")
    # os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2") or st.secrets.get("LANGCHAIN_TRACING_V2", "")
    # os.environ["LANGCHAIN_PROJECT"] = st.secrets.get("LANGCHAIN_PROJECT", "")
    # os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")
    # os.environ["SERPER_API_KEY"] = st.secrets.get("SERPER_API_KEY", "")
    # os.environ["FIRECRAWL_API_KEY"] = st.secrets.get("FIRECRAWL_API_KEY", "")
    # os.environ["LINKEDIN_SEARCH"] = st.secrets.get("LINKEDIN_JOB_SEARCH", "")

    # Setup directories and paths
    temp_dir = "temp"
    dummy_resume_path = os.path.abspath("dummy_resume.pdf")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    if not os.path.exists(dummy_resume_path):
        default_resume_path = "path/to/your/dummy_resume.pdf"
        shutil.copy(default_resume_path, dummy_resume_path)

    st.sidebar.title("Ascendita Toolbox")

    with st.sidebar.expander("Resume", expanded=True):
        uploaded_document = st.file_uploader("Upload Your Resume", type="pdf", key="resume_upload")
        if uploaded_document is not None:
            bytes_data = uploaded_document.read()
            filepath = os.path.join(temp_dir, "resume.pdf")
            with open(filepath, "wb") as f:
                f.write(bytes_data)
            st.markdown("**Resume uploaded successfully!**")
        else:
            st.write("Using a dummy resume for demonstration purposes.")
            st.markdown(
                "[View Dummy Resume](https://drive.google.com/file/d/1vTdtIPXEjqGyVgUgCO6HLiG9TSPcJ5eM/view?usp=sharing)",
                unsafe_allow_html=True
            )
            bytes_data = open(dummy_resume_path, "rb").read()
            filepath = os.path.join(temp_dir, "resume.pdf")
            with open(filepath, "wb") as f:
                f.write(bytes_data)

    with st.sidebar.expander("Provider Settings", expanded=True):
        global settings
        settings = None

        # Default order: Google → OpenAI → Groq
        service_provider = st.selectbox(
            "Choose AI Provider",
            ("google", "openai", "groq (llama-3.1-70b-versatile)"),
            key="provider_select"
        )

        if service_provider == "google":
            api_key_google = st.text_input(
                "Google API Key",
                st.session_state.get("GOOGLE_API_KEY", ""),
                type="password",
                key="google_api_key"
            )
            model_google = st.selectbox(
                "Google Model",
                ("gemini-1.5-flash",),
                key="google_model_select"
            )
            settings = {
                "model": "models/" + model_google,
                "model_provider": "google_genai",
                "temperature": 0.3,
            }
            st.session_state["GOOGLE_API_KEY"] = api_key_google
            os.environ["GOOGLE_API_KEY"] = st.session_state["GOOGLE_API_KEY"]

        elif service_provider == "openai":
            api_key_openai = st.text_input(
                "OpenAI API Key",
                st.session_state.get("OPENAI_API_KEY", ""),
                type="password",
                key="openai_api_key"
            )
            model_openai = st.selectbox(
                "OpenAI Model",
                ("gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"),
                key="openai_model_select"
            )
            settings = {
                "model": model_openai,
                "model_provider": "openai",
                "temperature": 0.3,
            }
            st.session_state["OPENAI_API_KEY"] = api_key_openai
            os.environ["OPENAI_API_KEY"] = st.session_state["OPENAI_API_KEY"]

        else:  # Default to Groq if selected
            if "groq_key_visible" not in st.session_state:
                st.session_state["groq_key_visible"] = False
            if st.button("Enter Groq API Key (optional)", key="groq_button"):
                st.session_state["groq_key_visible"] = True
            if st.session_state["groq_key_visible"]:
                api_key_groq = st.text_input("Groq API Key", type="password", key="groq_api_key")
                st.session_state["GROQ_API_KEY"] = api_key_groq
                os.environ["GROQ_API_KEY"] = api_key_groq
            settings = {
                "model": "llama-3.1-70b-versatile",
                "model_provider": "groq",
                "temperature": 0.3,
            }

    # Initialize the chat history and agent graph
    global message_history
    message_history = StreamlitChatMessageHistory()
    flow_graph = define_graph()

    if "active_option_index" not in st.session_state:
        st.session_state["active_option_index"] = None
    if "interaction_history" not in st.session_state:
        st.session_state["interaction_history"] = []
    if "response_history" not in st.session_state:
        st.session_state["response_history"] = ["Hello! How can I assist you today?"]
    if "user_query_history" not in st.session_state:
        st.session_state["user_query_history"] = []
    if "new_query" not in st.session_state:
        st.session_state["new_query"] = ""

    # Query form with two columns: left for input, right for query options.
    st.subheader("💬 Ask Ascendita")
    query_col_left, query_col_right = st.columns([2, 1])
    default_query = st.session_state.get("selected_query_option", "Identify top Gen AI trends in tech")
    with query_col_left:
        with st.form(key="query_form", clear_on_submit=True):
            user_input_query = st.text_input(
                "Query:",
                value=default_query,
                placeholder="📝 Write your query or select from the options",
                key="input"
            )
            submit_query_button = st.form_submit_button(label="Send")
    with query_col_right:
        query_options = [
            "Identify top Gen AI trends in tech",
            "Discover emerging opportunities in emerging tech",
            "Summarize and optimize my resume",
            "Visualize a career path based on my skills",
            "Find Gen AI jobs at leading companies",
            "Search for job listings using Gen AI",
            "Analyze my resume for suitable roles",
            "Generate a tailored cover letter",
        ]
        selected_query_option = st.selectbox("Query Options", query_options, key="query_options")
        st.session_state["selected_query_option"] = selected_query_option

    # Instead of calling execute_chat_conversation immediately (since we need a proper container),
    # we store the new query in session state.
    if submit_query_button and user_input_query:
        st.session_state["new_query"] = user_input_query

    # Chat Conversation expander with two columns: left for messages, right for Ascendita's Action.
    with st.expander("Chat Conversation", expanded=True):
        chat_left, chat_right = st.columns([2, 1])
        with chat_left:
            if st.button("Clear Chat", key="clear_chat_button"):
                st.session_state["user_query_history"] = []
                st.session_state["response_history"] = []
                message_history.clear()
                st.experimental_rerun()
            if st.session_state["response_history"]:
                for i in range(len(st.session_state["response_history"])):
                    if i < len(st.session_state["user_query_history"]):
                        message(
                            st.session_state["user_query_history"][i],
                            is_user=True,
                            key=str(i) + "_user",
                        )
                    message(
                        st.session_state["response_history"][i],
                        key=str(i),
                    )
        # with chat_right:
    st.subheader("⚡ Ascendita's Action")
    # If there's a new query pending, execute the chat conversation using chat_right as the output container.
    if st.session_state["new_query"]:
        chat_output = execute_chat_conversation(
            st.session_state["new_query"], flow_graph, chat_right
        )
        st.session_state["user_query_history"].append(st.session_state["new_query"])
        st.session_state["response_history"].append(chat_output)
        st.session_state["last_input"] = st.session_state["new_query"]
        st.session_state["active_option_index"] = None
        st.session_state["new_query"] = ""  # Clear the new query after processing

streamlit_analytics.stop_tracking()

if __name__ == "__main__":
    main()



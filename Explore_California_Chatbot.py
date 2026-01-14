#Build with AI: LLM-Powered Applications with Streamlit
#Deploy Your Chatbot to Streamlit Community Cloud for Free
 
#Import packages
import streamlit as st
from openai import OpenAI
import pandas as pd
import logging, time, traceback, os
#from langchain.docstore.document import Document
#from langchain.text_splitter import CharacterTextSplitter
#from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

#Gather API key
my_api_key = os.getenv("OPENAI_API_KEY")

#Initialize OpenAI client with your API key
client = OpenAI(api_key=my_api_key)

#Configure page
st.set_page_config(page_title="Chat UI")

#Write title
st.title("Tour Chatbot")

#Configure a logging file
logging.basicConfig(
    filename="chatbot_rag.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

#Cache global resources
@st.cache_resource
#Define vector store function
def load_vector_store():
    #Load California Tour Packages data
    df = pd.read_excel("Explore_California_Tour_Packages.xlsx")

    #Combine text fields into a single column for embedding
    df["combined_text"] = (
        df["Tour_Name"].fillna("") + ". " +
        df["Tour_Description"].fillna("") + ". " +
        df["Tour_Summary"].fillna("") + ". " +
        df["Tour_Keywords"].fillna("") + ". " +
        df["Tour_Itinerary"].fillna("") + "."
    ).str.strip()

    #Create lise of combined text entries, excluding empty strings
    text_data = df["combined_text"].loc[df["combined_text"] != ""].tolist()

    #Convert text entries into document objects
    docs = [Document(page_content=txt) for txt in text_data]
    #Initialize and run character-based text splitter
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    #Generate embeddings for each text chunk
    embeddings = OpenAIEmbeddings(openai_api_key=my_api_key)
    
    #Create and return FAISS vector store
    return FAISS.from_documents(chunks, embeddings)

#Create vector store from loaded and embedded documents
vector_store = load_vector_store()

#Determine if chat history exists in the session state and initialize if it doesn't
if "history" not in st.session_state:
    st.session_state.history = []

#Check if session state variable for tracking last answer index exists and initalize if not
if "last_answer_index" not in st.session_state:
    st.session_state.last_answer_index = None

#Define function to handle user's question and provide feedback
def send_and_feedback():
    #Retrieve user's query and remove leading/trailing whitespace
    user_query = st.session_state.query.strip()
    #Check is user query is empty
    if not user_query:
        #Provide warning if query is empty
        st.warning("Please enter a question about your tours above.")
        return

    #If query exists, add to chat history
    st.session_state.history.append(("You", user_query))
    #Display spinner animation while processing results
    with st.spinner("Retrieving relevant tour info…"):
        try:
            #Track start time for document search
            docs_start=time.time()
            #Perform similarity search in the vector store for the top 3 most relevant documents
            docs = vector_store.similarity_search(user_query, k=3)
            #Calculate elapsed time for document search
            docs_time = time.time() - docs_start
            #Log user query
            logging.info(f"Query: {user_query}")
            #Log number of documents found and time taken
            logging.info(f"Found {len(docs)} documents in {docs_time:.2f}s")

            if not docs:
                #Provide error message if no documents found
                st.info("No relevant documents found. Try rephrasing your question.")
                st.session_state.history.append(("Bot", "[No docs found]"))
                st.session_state.query = ""
                return

            #Combine retrieved text into a single context string
            context = "\n\n---\n\n".join([d.page_content for d in docs])
            #Provide a system prompt
            prompt = (
                "You are a helpful tour assistant. Use the context below to answer.\n\n"
                f"{context}\n\nQuestion: {user_query}"
            )
            #Track start time for OpenAI call
            api_start = time.time()
            #Send prompt to OpenAI chat completion API
            resp = client.chat.completions.create(
                #Select model
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            #Calculate elapsed time for API call
            api_time = time.time() - api_start
            #Gather assistant's response
            bot_reply = resp.choices[0].message.content.strip()
            #Log API response time
            logging.info(f"API response time: {api_time:.2f}s")
            #Log first 200 characters of assistant's reply
            logging.info(f"Answer: {bot_reply[:200]}...")

            #Add AI assistant's reply to chat history
            st.session_state.history.append(("Bot", bot_reply))
            #Update index of last answer
            st.session_state.last_answer_index = len(st.session_state.history) - 1
            #Show success message
            st.success("Response generated!")

        except Exception as e:
            #Handle A{I errors}
            st.error(f"Failed to get response: {e}")
            #Log error type and corresponding message
            logging.error(f"Error: {type(e).__name__}: {e}")
            #Log full traceback error
            logging.error(traceback.format_exc())
            #Add errpr message to chat history
            st.session_state.history.append(("Bot", "[Error generating response]"))
    #Clear text input field
    st.session_state.query = ""


#Create text input field that triggers send_and_feedback when enter button is pressed
st.text_input("Ask a question about tours:", key="query", on_change=send_and_feedback)

st.markdown("### Chat History")
#Loop through chat history and display messages
for i, (speaker, msg) in enumerate(st.session_state.history):
    if speaker == "You":
        st.write(f"**{speaker}:** {msg}")
    else:
        st.info(f"**{speaker}:** {msg}")
        #Provide feedback buttons for latest answer
        if i == st.session_state.last_answer_index:
            #Create two columns
            col1, col2 = st.columns(2)
            with col1:
                #Create "Helpful" user feedback button
                if st.button("👍 Helpful", key=f"up_{i}"):
                    logging.info("User feedback: 👍")
            with col2:
                #Create "Not Helpful" user feedback button
                if st.button("👎 Not Helpful", key=f"down_{i}"):
                    logging.info("User feedback: 👎")

st.markdown("### Recent Log Entries")
#Open chatbot log file
with open("chatbot_rag.log") as f:
    #Read last 10 lines of log file
    lines = f.readlines()[-10:]
#Display last 10 log entries as a code block

st.code("".join(lines))

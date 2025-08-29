import streamlit as st
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

#Setting the title and layout for the Streamlit page
st.set_page_config(page_title="Text to Math Problem Solver and Data Search Assistant")
st.title("Text to Math Problem Solver Using Gemma 2")

#Creating a sidebar input for the user to enter their Groq API key
groq_api_key = st.sidebar.text_input(label="GROQ API Key", type="password")

#Checking if the API key has been entered. If not, stop the app.
if not groq_api_key:
    st.info("Please add your API key to continue")
    st.stop()


#Initializing the LLM
llm = ChatGroq(model='gemma2-9b-it', groq_api_key=groq_api_key)

# Initializing tools
# Wikipedia Tool used for searching general knowledge questions
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="wikipedia",
    func=wikipedia_wrapper.run,
    description="Tool for Searching the Internet to find various information on the topics mentioned"
)
#Calculator Tool used for solving mathematical expressions
# LLMMathChain is specifically designed to handle math problem
math_chain = LLMMathChain.from_llm(llm=llm)
calculator_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Tool for answering math related questions. only input mathematical expression needs to be provided"
)

#Defining a Prompt Template for Reasoning
prompt = """
You are a agent tasked for solving mathematical questions.Logically arrive at the solution and provide a detailed explanation and display it point wise for the question below
Question:{question}
Answer:
"""
prompt_template = PromptTemplate(
    input_variables=['question'],
    template=prompt
)

#Combining tools into chain
chain = LLMChain(llm=llm, prompt=prompt_template)

#Reasoning Tool for handling logical and reasoning-based questions
reasoning_tool = Tool(
    name='Reasoning',
    func=chain.run,
    description="Tool for answering logic-based and reasoning question"
)

# initializing agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

#Initializing session state to store chat messages if it doesn't exist
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role': "assistant", "content": "Hi, I'm a Math chatbot who can answer all your math questions"}
    ]

#Displaying existing chat messages
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

#Creating a text input box for the user to ask a question, with a default example
question = st.text_input("Enter your question: ",
                         "I have 5 bananas and 7 grapes. I eat 2 bananas and give away 3 grapes. Then I buy a dozen apples and 2 packs of blueberries. Each pack of blueberries contains 25 berries. How many total pieces of fruit do I have at the end?")


if st.button("find my answer"):
    if question:
        with st.spinner("Generating response..."):
            #Adding the user's question to the chat history
            st.session_state.messages.append({"role": "user", "content": question})
            st.chat_message('user').write(question)

            #Using a callback handler to display the agent's thoughts in the Streamlit container
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            #Running the agent to get the final answer
            response = assistant_agent.run(question, callbacks=[st_cb])
            #Running the reasoning chain to get the step-by-step explanation
            reasoning = chain.run(question)

            final_answer = f"### Reasoning:\n{reasoning}\n\n### Final Answer:\n#### {response}"
            st.success(final_answer)

else:
    st.warning("Please enter the question")

import getpass
from dotenv import load_dotenv
import os
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter Api key for OPENAI")

import bs4
from langchain import hub
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict

loader = UnstructuredExcelLoader(
    "./red.xlsx",
    mode="elements"
)
docs = loader.load()

from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
from langchain_core.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)
#prompt = hub.pull("rlm/rag-prompt")

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# retrieve data from vector_store and handle over next tool `generate` with key named `context`
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}
def generate(state: State):
    if "context" in state and state["context"]:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    else:
        print("state['context'] not exists")
        docs_content = ""
    messages = custom_rag_prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

#from PIL import Image
#try:
#    d = graph.get_graph().draw_mermaid_png()
#    open('output.png', 'wb').write(d)
#    img = Image.open('output.png')
#    img.show()
#except Exception:
#    # This requires some extra dependencies and is optional
#    print(f"image exception:{Exception.with_traceback()}")
#
import json
from langchain_core.load import dumps
def stream_graph_updates(user_input: str):
    for event in graph.stream ({"question": user_input}):
        if "generate" in event:
            print("Assistant:", event["generate"]["answer"])
#        json_event = dumps(event, pretty=True)
#        with open('data.json', 'w', encoding='utf-8') as f:
#            json.dump(json_event, f)

while True:
    try:
        user_input = input("you: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except Exception as e:
        print(f"An error occurred: {e}")
#response = graph.invoke({"question": "최현겸에 대해서 말해줘?"})
#print(response["answer"])

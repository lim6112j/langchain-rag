from langchain_core.load import dumps
import json
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain import hub
import bs4
import getpass
from dotenv import load_dotenv
import os
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter Api key for OPENAI")


loader = UnstructuredExcelLoader(
    "./red.xlsx",
    mode="elements"
)
docs = loader.load()

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = InMemoryVectorStore(embeddings)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering

template = """You are a Whole Sale manager who sells internet service and mobile subscription.
you should calculate your sub saler commissions from sales report excel file.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
`SKB정책` sheet contains the commision calculation rules when you calculate sub-salers commission.
`하부유통망정산` sheet contains the sub-salers sale results for a given period.
With `하부유통망정산` sheet and `정책` sheet, you can calculate the commision of sub-saler.
{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)
# prompt = hub.pull("rlm/rag-prompt")


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
        docs_content = "\n\n".join(
            doc.page_content for doc in state["context"])
    else:
        print("state['context'] not exists")
        docs_content = ""
    messages = custom_rag_prompt.invoke(
        {"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# from PIL import Image
# try:
#    d = graph.get_graph().draw_mermaid_png()
#    open('output.png', 'wb').write(d)
#    img = Image.open('output.png')
#    img.show()
# except Exception:
#    # This requires some extra dependencies and is optional
#    print(f"image exception:{Exception.with_traceback()}")
#


def stream_graph_updates(user_input: str):
    for event in graph.stream({"question": user_input}):
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
# response = graph.invoke({"question": "최현겸에 대해서 말해줘?"})
# print(response["answer"])

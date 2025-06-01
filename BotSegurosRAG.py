
import os
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import init_chat_model
import streamlit as st

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
llm = init_chat_model("gpt-3.5-turbo",model_provider = "openai")



embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embeddings)

from langchain_community.document_loaders import PyMuPDFLoader


# Ruta del PDF en tu PC
pdf_path = "Preguntas_Respuestas.pdf"  # Reemplaza con la ruta real

# Cargar contenido del PDF
loader = PyMuPDFLoader(pdf_path)
docs = loader.load()

# Unir todo el texto en un solo string
full_text = "\n\n".join([doc.page_content for doc in docs])

# Crear un único documento
from langchain_core.documents import Document  
docs = [Document(page_content=full_text)]

print(f"Cantidad de documentos cargados: {len(docs)}")

print(f"Número de documentos: {len(docs)}")
#print(f"Contenido del documento:\n{docs[0].page_content[:500]}...")  # Mostrar primeras 500 caracteres

print(f"Total characters: {len(docs[0].page_content)}")
#assert len(docs) == 1
#print(f"Total characters: {len(docs[0].page_content)}")
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

document_ids = vector_store.add_documents(documents=all_splits)

print(document_ids[:3])
#Prompt de contexto
from langchain_core.prompts import PromptTemplate

custom_prompt = PromptTemplate.from_template(
    """
    Eres un asistente experto en análisis de documentos.
    Tienes Acceso a un documento en ingles, que contiene informacion de unas preguntas, y debajo de cada una de estas
    su respectiva respuesta usa esta informacion para el contexto de la respuesta 
    Tienes acceso a un conjunto de documentos que contienen información detallada sobre un tema.
    Usa TODA la información relevante en el contexto para responder de la forma más precisa posible.
    la respuesta debe ser la que tiene la pregunta en el documento y en ingles como esta el documento

    Pregunta del usuario: {question}

    Contexto relevante del documento:
    {context}
    
    Responde basado solamente en el documento, si la pregunta esta, debes dar la respuesta asociada debajo, si tiene mas de una pues responde 
    con ambas, ademas de eso si no esta la pregunta en el documento, debes informarme y darme la respuesta que tu creerias posible pero siempre
    informandome que esa pregunta no estaba en el documento
    """
)


##
#from langchain import hub
#prompt = hub.pull("rlm/rag-prompt")
##

example_messages = custom_prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()

assert len(example_messages) == 1
print(example_messages[0].content)

from langchain_core.documents import Document
from typing_extensions import List, TypedDict


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = custom_prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

from langgraph.graph import START, StateGraph
import streamlit as st

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


##Imagen de respuesta
#from IPython.display import Image, display
#display(Image(graph.get_graph().draw_mermaid_png()))

##Front de pregunta
st.title("🤖 Agente Experto en Automation360 💵")

# Crear una caja de entrada de texto
question = st.text_input("Escribe tu pregunta aquí:")

#boton para ejecutar
if st.button("Preguntar"):
    if question:
        with st.spinner("buscando respuesta..."):
            result = graph.invoke({"question": question})
            st.write("#### Respuesta:")
            st.write(result["answer"])
    else:
        st.warning("Por favor, ingresa una pregunta")
from dotenv import load_dotenv

load_dotenv()

from knowledge_base import retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.documents.base import Document
from langgraph.graph.state import CompiledStateGraph


@tool
def get_context(query: str) -> tuple[str, list[Document]]:
    """Retrieve information to help answer a query."""
    retrieved_docs = retriever.invoke(query)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# ## Agent Prompt


prompt = """
You are a helpful Clerical assistant. You have access to the following tool:
`get_context`: This can be used if you need to find any file information for the user's question. 
It takes in a search parameter 'q' as input and this is used to perform vector search in the vector database having file information.
The tool returns a tuple containing information text having content, file name and document metadata information.

< Answer >
Respond to the user question in a friendly manner. If its a general question and if you know the answer, feel free to answer and insist the user that you can help with clerical information.
In case you have used the `get_context` tool and it was helpful, let the user know you have searched for the files and found the most appropriate information in the file 'foo'
</Answer>
"""


# ## Agent creation


agent: CompiledStateGraph = create_agent(
    model=ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite"),
    tools=[get_context],
    system_prompt=prompt,
)

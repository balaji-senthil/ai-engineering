from dotenv import load_dotenv

load_dotenv()

from knowledge_base import retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, MessagesState
from typing import Literal
from pydantic import BaseModel, Field
from langchain.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# ## Tool definition


from langchain_classic.tools.retriever import create_retriever_tool

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_docs",
    "Search and return information about inventory invoices",
)


# ### Nodes


router_llm_instructions: str = """You are an expert Classification assistant. From the given user query, classify if it is related to inventory management or not If it is not related, give a straight up answer for the question.
Question: {question}
"""


def router_node(state: MessagesState) -> dict[str, object]:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}


# ### Question rewriter for better retrieval


class ReWriterLLMResponse(BaseModel):
    """Updated Rewritten Question Model"""

    updated_question: str = Field(description="Updated Question")


def re_write_question(state: MessagesState) -> dict[str, object]:
    """Question rewriting for better retrieval"""
    messages = state["messages"]
    question = messages[0].content
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)
    res: ReWriterLLMResponse = llm.with_structured_output(ReWriterLLMResponse).invoke(
        f"""
You are a helpful question rewriter. The rewritten question will be used to retrieve documents from a vector store.
Current question: {question}
"""
    )
    return {"messages": [HumanMessage(content=res.updated_question)]}


# ### Grade docs


class GraderResponse(BaseModel):
    """Response of an Grader assistant that gives a confidence score of a given user question to the given document content"""

    confidence: float = Field(
        description="Confidence score of the document with relevance to user question",
        le=1.0,
        ge=0.0,
    )


def grader_node(
    state: MessagesState,
) -> Literal["re_write_question", "generate_answer"]:
    router_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    grader = router_llm.with_structured_output(GraderResponse)
    question = state["messages"][0].content
    context = state["messages"][-1].content
    res = grader.invoke(
        [
            {
                "role": "user",
                "content": f"""
    You are an expert Grader assistant. From the given user query, grade the given document content. The scores can range from 0.0 to 1.0. 
    0.0 being no confidence and 1.0 being fully confident.

    Question: {question}
    Document: {context}
    """,
            }
        ]
    )
    if res.confidence > 0.8:
        return "generate_answer"

    return "re_write_question"


# ### Generate answer node


class FinalResponse(BaseModel):
    answer: str = Field(description="Final answer for the user question")


def generate_answer(state: MessagesState) -> dict[str, list[str]]:
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0)
    question = state["messages"][0].content
    context = state["messages"][-1].content
    res: FinalResponse = llm.with_structured_output(FinalResponse).invoke(
        f"""
You are a helpful Clerical assistant. For the given user question and supporting documents, generate a response.

< Answer >
Respond to the user question in a friendly manner.
let the user know you have searched for the files and found the most appropriate information:

Question: {question}
Context: {context}
Answer: ?
</Answer>
"""
    )
    return {"messages": [res.answer]}


# ## Graph


workflow = StateGraph(MessagesState)

workflow.add_node(router_node)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(re_write_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "router_node")

workflow.add_conditional_edges(
    "router_node",
    tools_condition,
    {
        "tools": "retrieve",
        END: END,
    },
)

workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grader_node,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("re_write_question", "router_node")

# Compile
graph = workflow.compile()

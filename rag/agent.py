"""Retrieve evidence from SEC filings and answer questions with citations.

A decision model chooses the initial search and judges the retrieved evidence.
Chat models extract financial concepts, write search plans, and compose answers.
The retrieval graph supports text searches, stronger searches, and plans for
missing evidence; figure questions use local financial tables. The app exposes
this workflow through a conversational agent with two tools.
"""
import json
import operator
import os
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

from . import config, decide as decision, retrieval
from .util import retry, text_of

MAX_ATTEMPTS = 3

# What each kind of question starts with. Measured in notebook 04 of the
# ladder version: the filter wins on specific questions and loses on
# comparative ones, where hybrid wins.
START_WITH = {"specific": "filter", "comparative": "hybrid+rrf", "figure": "filter"}
DEFAULT = "filter+rrf"            # when the router is not sure
SURE = 0.6                        # below this, use DEFAULT

# wrong_words means the search was aimed right and missed, so search harder.
STRONGER = {"filter": "filter+rrf", "hybrid+rrf": "filter+rrf+rerank",
            "filter+rrf": "filter+rrf+rerank"}

# absent is a judgment, and the model can be unsure of it. Below this we
# give it one more search before taking its word.
ABSENT_SURE = 0.8


# ---------------------------------------------------------------- the tools

def financial_fact(ticker: str, concept: str) -> str:
    """Read a reported financial figure from a company's local JSON file.

    ticker must be NVDA, AMD, or MU. concept is an XBRL name such as
    Revenues or ResearchAndDevelopmentExpense. Return the available values
    with their period-ending dates. If the company or concept is missing,
    return an explanation instead of performing a text search."""
    path = os.path.join(config.FINANCIALS, f"{ticker.upper()}_financials.json")
    if not os.path.exists(path):
        return f"No financials for {ticker}. Use NVDA, AMD or MU."
    with open(path) as file:
        facts = json.load(file)["facts"]
    if concept not in facts:
        return (f"{ticker} has no concept named {concept}. Available: "
                + ", ".join(sorted(facts)[:25]))
    rows = sorted(facts[concept].items())
    return f"{ticker} {concept}:\n" + "\n".join(
        f"  period ending {d}: {v:,}" for d, v in rows)


def format_passages(hits):
    """Format (source, text) pairs as passages with bracketed source names.

    The resulting text is used in prompts so the answer model can cite the
    filing associated with each passage."""
    return "\n\n".join(f"[{source}] {text}" for source, text in hits)


# ---------------------------------------------------------------- the state

class RetrievalState(TypedDict, total=False):
    """State shared by the retrieval graph's nodes.

    Track the question, chosen strategy, evidence, diagnosis, search attempts,
    model-call counts, and final answer. The documents and call-count fields
    accumulate updates; other fields are replaced when a node returns them."""
    question: str
    scope: str                 # specific, comparative, figure
    strategy: str              # which retrieval to run next
    documents: Annotated[list, operator.add]
    verdict: str               # sufficient, absent, wrong_words, incomplete
    probabilities: dict
    steps: list
    attempts: int
    chat_calls: Annotated[int, operator.add]
    decision_calls: Annotated[int, operator.add]
    answer: str


# ---------------------------------------------------------------- the nodes

def route(state: RetrievalState) -> dict:
    """Classify the question and choose its starting retrieval strategy.

    Ask the decision model for a scope and its probabilities. Use START_WITH when the selected 
    scope meets SURE; otherwise use DEFAULT. Return the scope and strategy, and record one decision-model call."""
    # TODO 3.1
    # your code here
    scope, probs = retry(decision.scope, state["question"])
    if probs[scope] >= SURE:
        strategy = START_WITH[scope]
    else:
        strategy = DEFAULT
    return {"scope": scope, "strategy": strategy, "decision_calls": 1}

def after_route(state: RetrievalState) -> Literal["lookup", "retrieve"]:
    """Choose the next node from the question's scope.
    Send figure questions to lookup, which reads structured financial data.
    Send all other questions to retrieve, which searches filing passages."""
    # TODO 3.1
    # your code here
    if state["scope"] == "figure":
        return "lookup"
    else:
        return "retrieve"

class Fact(BaseModel):
    """Structured output identifying the company and financial concept to read.

    The allowed tickers and XBRL concepts match the financial lookup tool.
    Period selection is handled when answering from the returned table."""
    ticker: Literal["NVDA", "AMD", "MU"]
    concept: Literal["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                     "ResearchAndDevelopmentExpense", "GrossProfit", "NetIncomeLoss",
                     "OperatingIncomeLoss"] = Field(
        description="the XBRL concept. Revenue questions: Revenues, or "
                    "RevenueFromContractWithCustomerExcludingAssessedTax for AMD. "
                    "R&D spend: ResearchAndDevelopmentExpense.")


FACT_PROMPT = """Which reported figure does this question ask for?

  research and development spend  -> ResearchAndDevelopmentExpense
  revenue, sales, top line         -> Revenues (AMD reports it as
                                      RevenueFromContractWithCustomerExcludingAssessedTax)
  gross profit, gross margin       -> GrossProfit
  net income, profit, earnings     -> NetIncomeLoss
  operating income                 -> OperatingIncomeLoss

ticker is the company's symbol: NVIDIA -> NVDA, AMD -> AMD, Micron -> MU.

Example: "What did Micron spend on R&D in fiscal 2025?"
         -> ticker=MU, concept=ResearchAndDevelopmentExpense

Question: {question}"""


def lookup(state: RetrievalState) -> dict:
    """Identify the requested financial concept and read its reported values.

    Use one structured chat-model call to obtain a Fact, then call
    financial_fact locally. Return the table as a source-tagged document,
    mark the verdict sufficient, and record one attempt and one chat call.
    The graph then goes directly to the answer node."""
    # TODO 3.2
    # your code here
    namer = config.chat().with_structured_output(Fact)
    fact = retry(namer.invoke, FACT_PROMPT.format(question=state["question"]))
    table = financial_fact(fact.ticker, fact.concept)
    return {"documents": [(f"{fact.ticker}_financials.json", table)], "verdict": "sufficient", "attempts": 1, "chat_calls": 1}

def retrieve(state: RetrievalState) -> dict:
    """Search with the current strategy and return newly found passages.

    Exclude passages already in state because the documents reducer appends
    this update to existing evidence. Increment attempts and record the chat
    calls used by the selected retrieval strategy."""
    search = retrieval.STRATEGIES[state["strategy"]]
    hits = search(state["question"], config.K)
    seen = set(state.get("documents") or [])
    new = [h for h in hits if h not in seen]
    chat = {"filter": 1, "hybrid+rrf": 1, "filter+rrf": 2, "filter+rrf+rerank": 2}
    return {"documents": new, "attempts": state.get("attempts", 0) + 1,
            "chat_calls": chat[state["strategy"]]}


def diagnose(state: RetrievalState) -> dict:
    """Judge whether the accumulated passages answer the question.

    Return the decision model's verdict and probabilities, counting one decision call. 
    If an absent verdict is below ABSENT_SURE and attempts remain, 
    treat it as wrong_words so another search can be considered. Keep the original probabilities even when the verdict is changed."""
    # TODO 3.3
    # your code here
    verdict, probs = retry(decision.diagnose, state["question"], state.get("documents", []))
    if verdict == "absent" and probs["absent"] < ABSENT_SURE and state.get("attempts", 0) < MAX_ATTEMPTS:
        verdict = "wrong_words"
    return {"verdict": verdict, "probabilities": probs, "decision_calls": 1}

def decide(state: RetrievalState) -> Literal["answer", "stronger", "plan"]:
    """Choose whether to answer, search harder, or plan missing searches.

    Stop for sufficient or absent verdicts, or when MAX_ATTEMPTS is reached. An incomplete verdict goes to plan. 
    A wrong_words verdict goes to stronger when STRONGER contains a next strategy; otherwise answer with
    the evidence collected so far."""
    # TODO 3.3
    # your code here
    if state["verdict"] in ("sufficient", "absent"):
        return "answer"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "answer"
    if state["verdict"] == "incomplete":
        return "plan"
    if state["strategy"] in STRONGER:
        return "stronger"
    return "answer"

def stronger(state: RetrievalState) -> dict:
    """Replace the current retrieval strategy with the next entry in STRONGER.

    The graph calls this node only when a stronger strategy is available,
    then returns to retrieve with the same original question."""
    # TODO 3.4
    # your code here
    return {"strategy": STRONGER[state["strategy"]]}

class Plan(BaseModel):
    """Structured output containing the separate searches still needed.

    Each step should name one company and one topic, with a fiscal year when
    relevant, so a filtered search can target that part of the question."""
    steps: list[str] = Field(description="Each names one company, one year if "
                                         "relevant, and one thing to find")


PLAN_PROMPT = """A question about SEC 10-K filings needs more than one search:

{question}

Passages found so far cover: {covered}

Write the separate questions still needed. Each must name exactly one company
and ask for one thing, so that a filtered search can find it. Fewest possible."""


def plan(state: RetrievalState) -> dict:
    """Ask the chat model to plan searches for missing parts of the question.

    Include the source filenames already collected in PLAN_PROMPT so the planner knows 
    which filings have been retrieved. Return the proposed steps and record one chat call. This node does not run the searches."""
    # TODO 3.4
    # your code here
    planner = config.chat().with_structured_output(Plan)
    sources = sorted({source for source, _ in state.get("documents", [])})
    if sources:
        covered = ", ".join(sources)
    else:
        covered = "nothing"
    out = retry(planner.invoke, PLAN_PROMPT.format(question=state["question"], covered=covered))
    return {"steps": out.steps, "chat_calls": 1}

def execute(state: RetrievalState) -> dict:
    """Run a filtered search for each step in the current plan.

    Return only passages not already in state or collected earlier in this execution. 
    The documents reducer appends them to existing evidence. Count the whole plan execution as one attempt and each filter extraction
    as one chat call, then let the graph diagnose the combined evidence."""
    # TODO 3.4
    # your code here
    seen = set(state.get("documents", []))
    documents = []
    steps = state.get("steps", [])
    for step in steps:
        hits = retrieval.with_filter(step, config.K)
        for hit in hits:
            if hit not in seen:
                documents.append(hit)
                seen.add(hit)
    return {"documents": documents, "attempts": state.get("attempts", 0) + 1, "chat_calls": len(steps)}

ANSWER_PROMPT = """Answer the question using only these passages from SEC
filings. Cite the filing each fact comes from, by its filename in brackets.
If the passages do not cover the question, say so in one sentence.

Question: {question}

Passages:
{passages}"""


def answer(state: RetrievalState) -> dict:
    """Produce an answer supported by the collected evidence.

    For an absent verdict, return the fixed unsupported-topic response with
    no chat call. Otherwise ask the chat model to answer from the passages,
    cite source filenames, and acknowledge missing support. Record one
    chat call for that generated answer."""
    if state.get("verdict") == "absent":
        return {"answer": "The filings in this corpus do not cover that.", "chat_calls": 0}
    out = retry(config.chat().invoke, ANSWER_PROMPT.format(
        question=state["question"], passages=format_passages(state["documents"])))
    return {"answer": text_of(out), "chat_calls": 1}


def build_graph():
    """Build and compile the retrieval workflow.

    Route figure questions through lookup to answer. Other questions pass through retrieval and diagnosis, 
    which can lead to a stronger search, a plan and its execution, or an answer. Conditional routing functions
    declare the possible destinations with their Literal return types."""
    # TODO 3.4
    # your code here
    from langgraph.graph import StateGraph, START, END

    graph = StateGraph(RetrievalState)
    graph.add_node("route", route)
    graph.add_node("lookup", lookup)
    graph.add_node("retrieve", retrieve)
    graph.add_node("diagnose", diagnose)
    graph.add_node("stronger", stronger)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("answer", answer)
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", after_route)
    graph.add_edge("lookup", "answer")
    graph.add_edge("retrieve", "diagnose")
    graph.add_conditional_edges("diagnose", decide)
    graph.add_edge("stronger", "retrieve")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "diagnose")
    graph.add_edge("answer", END)
    return graph.compile()

def run(question):
    """Run the retrieval graph for one question and return its final state.

    Initialize attempts and model-call counts to zero. The returned state
    includes the answer, evidence, final strategy and verdict, and counters
    used for tracing and evaluation."""
    return build_graph().invoke({"question": question, "attempts": 0,
                                 "chat_calls": 0, "decision_calls": 0})


def ask(question, verbose=False):
    """Return the retrieval graph's answer for one question.

    If verbose is True, also print the selected scope, final strategy,
    verdict, attempt count, and model-call counts."""
    state = run(question)
    if verbose:
        print(f"  {state['scope']} -> {state['strategy']}, verdict {state['verdict']}, "
              f"{state['attempts']} attempt(s), {state['chat_calls']} chat + "
              f"{state['decision_calls']} decision calls")
    return state["answer"]


# ---------------------------------------------------------------- the agent
#
# The graph above is the retrieval. The agent is what app.py serves: the same
# create_agent from W03b with two tools, the graph behind search_filings and
# the financials file behind financial_fact, and a checkpointer so the chat
# remembers the conversation. agentui.py streams it, tool calls and all.

def search_filings(question: str) -> str:
    """Answer a prose question using NVIDIA, AMD, and Micron 10-K filings.

    Search the business, risk-factor, and management-discussion sections
    through the retrieval graph. Return an answer with bracketed source
    filenames. Use financial_fact instead when the user wants a reported
    figure such as revenue or research and development spending."""
    return run(question)["answer"]


SYSTEM_PROMPT = """You answer questions about NVIDIA, AMD and Micron from their
10-K filings. Use search_filings for anything the filings say in prose, and
financial_fact for a reported number. Relay the tool's answer, with its
citations, and do not add facts of your own. If the tools find nothing, say so."""


def build_agent():
    """Create the conversational agent served by the app.

    Provide search_filings and financial_fact as tools and use SYSTEM_PROMPT
    to require grounded answers and citations. The memory checkpointer
    preserves conversation state for requests using the same thread ID."""
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import MemorySaver

    return create_agent(config.chat(), tools=[search_filings, financial_fact],
                        system_prompt=SYSTEM_PROMPT, checkpointer=MemorySaver())

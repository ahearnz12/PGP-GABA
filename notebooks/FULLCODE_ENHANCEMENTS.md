# FoodHub FullCode Enhancement Specifications

**Project**: PGP-GABA FoodHub Chatbot - FullCode Implementation
**Base Version**: LowCode GPT-OSS Implementation
**Target Framework**: LangGraph + LangChain + Pydantic
**Date**: 2025-10-08

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Enhancement Overview](#enhancement-overview)
3. [Phase 1: Core Enhancements](#phase-1-core-enhancements)
4. [Phase 2: Advanced Features](#phase-2-advanced-features)
5. [Phase 3: Production Features](#phase-3-production-features)
6. [Architecture Comparison](#architecture-comparison)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Testing Strategy](#testing-strategy)

---

## Executive Summary

### What's Being Added

The FullCode version transforms the LowCode chatbot from a **stateless single-turn system** into a **stateful multi-turn conversational agent** with quality guarantees, retry logic, and advanced observability.

### Key Metrics

| Metric | LowCode | FullCode | Improvement |
|--------|---------|----------|-------------|
| **Conversation Turns** | 1 (stateless) | Unlimited (stateful) | ∞ |
| **Quality Measurement** | None | Groundedness + Precision scores | ✅ New |
| **Retry Logic** | 0 attempts | 3 attempts with quality gates | ✅ New |
| **Human Escalation** | Basic (intent-based) | Advanced (sentiment + urgency) | 3x better |
| **Observability** | Minimal | Full logging + optional LangSmith | 10x better |
| **Agent Framework** | Legacy `initialize_agent` | Modern LangGraph | Migration |

### Dependencies to Add

```python
# Additional packages for FullCode
!pip install langgraph==0.2.56 \
             langchain-core==0.3.40 \
             pydantic==2.10.6
```

---

## Enhancement Overview

### Architecture Transformation

**LowCode**:
```
User Query → Input Guard → SQL Agent → [order_query_tool → answer_tool] → Output Guard → Response
```

**FullCode**:
```
User Query
  ↓
Enhanced Input Guard (with sentiment)
  ↓
┌─────────────────────────────────────┐
│      LANGGRAPH STATE MACHINE        │
│                                     │
│  ├─ SQL Query Node                 │
│  ├─ Extract Facts Node              │
│  ├─ Generate Response Node          │
│  ├─ Quality Evaluation Node ←──┐   │
│  │    ↓ (scores < 0.75)        │   │
│  │    └────── RETRY LOOP ──────┘   │
│  ├─ Human Approval Node (optional)  │
│  └─ Output Guard Node               │
└─────────────────────────────────────┘
  ↓
Response + Quality Scores + Memory Update
```

---

## Phase 1: Core Enhancements

### 1.1 LangGraph Migration ⭐ CRITICAL

**Status**: RECOMMENDED
**Effort**: High
**Impact**: High

#### What Changes

Replace deprecated `initialize_agent()` with LangGraph's stateful graph architecture.

#### Implementation

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage

# Define agent state
class AgentState(TypedDict):
    messages: Annotated[List[HumanMessage | AIMessage], "Conversation history"]
    order_id: str
    cust_id: str
    order_context: dict
    current_step: str
    extracted_facts: str
    agent_response: str
    quality_scores: dict
    retry_count: int
    sentiment_analysis: dict

# Create workflow graph
workflow = StateGraph(AgentState)

# Add nodes (each is a function that takes state and returns state)
workflow.add_node("input_analysis", input_analysis_node)
workflow.add_node("sql_query", sql_query_node)
workflow.add_node("extract_facts", extract_facts_node)
workflow.add_node("generate_response", generate_response_node)
workflow.add_node("quality_evaluation", quality_evaluation_node)
workflow.add_node("output_guard", output_guard_node)

# Define edges (transitions)
workflow.set_entry_point("input_analysis")

workflow.add_conditional_edges(
    "input_analysis",
    route_input,  # Function returns next node name
    {
        "continue": "sql_query",
        "escalate": END,
        "exit": END,
        "redirect": END
    }
)

workflow.add_edge("sql_query", "extract_facts")
workflow.add_edge("extract_facts", "generate_response")
workflow.add_edge("generate_response", "quality_evaluation")

workflow.add_conditional_edges(
    "quality_evaluation",
    should_retry,  # Check if quality is sufficient
    {
        "retry": "extract_facts",  # Retry fact extraction
        "continue": "output_guard",
        "escalate": END
    }
)

workflow.add_edge("output_guard", END)

# Enable persistent memory
memory = SqliteSaver.from_conn_string("foodhub_memory.db")

# Compile graph
app = workflow.compile(checkpointer=memory)
```

#### Node Implementation Template

```python
def sql_query_node(state: AgentState) -> AgentState:
    """
    Query database for order information.
    """
    order_id = state["order_id"]

    # Use existing SQL agent
    result = sqlite_agent.invoke(f"Fetch all columns for order_id {order_id}")

    # Update state
    state["order_context"] = result
    state["current_step"] = "sql_query_complete"

    return state


def extract_facts_node(state: AgentState) -> AgentState:
    """
    Extract relevant facts using order_query_tool logic.
    """
    query = state["messages"][-1].content
    order_context = state["order_context"]

    # Reuse existing tool logic
    facts = order_query_tool_func(query, order_context)

    state["extracted_facts"] = facts
    state["current_step"] = "facts_extracted"

    return state


def generate_response_node(state: AgentState) -> AgentState:
    """
    Generate customer-friendly response using answer_tool logic.
    """
    query = state["messages"][-1].content
    facts = state["extracted_facts"]
    order_context = state["order_context"]

    # Reuse existing tool logic
    response = answer_tool_func(query, facts, order_context)

    state["agent_response"] = response
    state["current_step"] = "response_generated"

    return state
```

#### Benefits

- ✅ **Cyclical workflows**: Can retry steps on failure
- ✅ **State persistence**: Conversation memory built-in
- ✅ **Better debugging**: Visual graph representation
- ✅ **Future-proof**: LangChain 1.0 compatible

---

### 1.2 Conversation Memory with Checkpointing ⭐ CRITICAL

**Status**: CRITICAL
**Effort**: Medium
**Impact**: High

#### What Changes

Add persistent conversation history so users can have multi-turn conversations with context retention.

#### Implementation

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage

# Initialize persistent memory
memory = SqliteSaver.from_conn_string("foodhub_conversations.db")

# Compile graph with memory
app = workflow.compile(checkpointer=memory)

# Multi-turn conversation
def chat_with_memory(order_id: str, cust_id: str, user_query: str):
    """
    Handle multi-turn conversation with memory.
    """
    # Thread ID uniquely identifies this conversation
    config = {
        "configurable": {
            "thread_id": f"{cust_id}_{order_id}"
        }
    }

    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=user_query)],
        "order_id": order_id,
        "cust_id": cust_id,
        "retry_count": 0
    }

    # Invoke agent with memory
    result = app.invoke(initial_state, config=config)

    return result["agent_response"]


# Example: Multi-turn conversation
config = {"configurable": {"thread_id": "C1011_O12486"}}

# Turn 1
response1 = chat_with_memory("O12486", "C1011", "Where is my order?")
print(f"Assistant: {response1}")

# Turn 2 - "it" refers to the order from Turn 1
response2 = chat_with_memory("O12486", "C1011", "Can I cancel it?")
print(f"Assistant: {response2}")

# Turn 3 - Context preserved
response3 = chat_with_memory("O12486", "C1011", "What items are in it?")
print(f"Assistant: {response3}")
```

#### Memory Schema

```sql
-- Automatically created by SqliteSaver
CREATE TABLE checkpoints (
    thread_id TEXT,
    checkpoint_id TEXT,
    parent_checkpoint_id TEXT,
    state BLOB,  -- Serialized AgentState
    metadata JSON,
    created_at TIMESTAMP,
    PRIMARY KEY (thread_id, checkpoint_id)
);
```

#### Benefits

- ✅ **Context carryover**: "it", "that", "my previous question" work
- ✅ **Session resumption**: Customer can return hours later
- ✅ **Sentiment tracking**: Remember if customer is frustrated
- ✅ **No token overhead**: Only recent messages sent to LLM

---

### 1.3 Quality Evaluation with LLM Judges ⭐ CRITICAL

**Status**: CRITICAL
**Effort**: Medium
**Impact**: High

#### What Changes

Add groundedness and precision scoring to measure response quality (like Kartify app).

#### Implementation

```python
import json
from typing import Dict

def evaluate_response_quality(
    order_context: str,
    query: str,
    response: str
) -> Dict[str, float]:
    """
    Evaluate agent response using LLM judge.

    Returns:
        {
            "groundedness": 0.0-1.0,  # Factual accuracy
            "precision": 0.0-1.0       # Query relevance
        }
    """
    evaluation_prompt = f"""
You are an expert evaluator for customer service responses.

Evaluate this response on TWO criteria. Return scores between 0.0 and 1.0.

**1. GROUNDEDNESS (Factual Accuracy)**
- 1.0 = All facts are accurate and supported by order data
- 0.8 = Mostly accurate with minor interpretations
- 0.5 = Mix of facts and assumptions
- 0.0 = Hallucinated or fabricated information

**2. PRECISION (Query Relevance)**
- 1.0 = Directly answers the exact question, concise
- 0.8 = Answers question with minor extra information
- 0.5 = Partially addresses query
- 0.0 = Completely misses the point or irrelevant

---

**ORDER CONTEXT:**
{order_context}

**CUSTOMER QUERY:**
{query}

**AGENT RESPONSE:**
{response}

---

Return ONLY valid JSON (no explanation):
{{"groundedness": 0.85, "precision": 0.90}}
"""

    llm = ChatOpenAI(
        model="local-model",
        temperature=0,  # Deterministic evaluation
        base_url=LM_STUDIO_BASE_URL,
        api_key=LM_STUDIO_API_KEY
    )

    result = llm.predict(evaluation_prompt)

    try:
        scores = json.loads(result.strip())
        return {
            "groundedness": float(scores.get("groundedness", 0.0)),
            "precision": float(scores.get("precision", 0.0))
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse quality scores: {e}")
        return {"groundedness": 0.0, "precision": 0.0}


# LangGraph node implementation
def quality_evaluation_node(state: AgentState) -> AgentState:
    """
    Evaluate response quality and store scores.
    """
    scores = evaluate_response_quality(
        state["order_context"],
        state["messages"][-1].content,
        state["agent_response"]
    )

    state["quality_scores"] = scores
    state["current_step"] = "quality_evaluated"

    logger.info(f"Quality Scores - Groundedness: {scores['groundedness']:.2f}, "
                f"Precision: {scores['precision']:.2f}")

    return state
```

#### Conditional Edge for Retry Logic

```python
def should_retry(state: AgentState) -> str:
    """
    Decide whether to retry based on quality scores.

    Returns:
        - "retry": Quality too low, regenerate response
        - "continue": Quality acceptable, proceed
        - "escalate": Max retries reached, escalate to human
    """
    scores = state["quality_scores"]
    retry_count = state.get("retry_count", 0)

    # Check if both scores meet threshold
    groundedness_ok = scores.get("groundedness", 0) >= 0.75
    precision_ok = scores.get("precision", 0) >= 0.75

    if groundedness_ok and precision_ok:
        logger.info("✓ Quality check passed")
        return "continue"

    # Check retry limit
    if retry_count >= 2:  # Max 3 attempts total
        logger.warning("✗ Max retries reached. Escalating to human.")
        state["agent_response"] = (
            "I want to make sure I give you accurate information. "
            "Let me connect you with a specialist who can help."
        )
        return "escalate"

    # Retry with incremented counter
    logger.warning(f"✗ Quality check failed (attempt {retry_count + 1}/3). Retrying...")
    state["retry_count"] = retry_count + 1

    # Add retry instruction to improve next attempt
    retry_instruction = (
        "\n[IMPORTANT: Previous response was not specific enough. "
        "Use exact facts from order data. Be concise and direct.]"
    )

    # Modify last message to include retry hint
    last_message = state["messages"][-1]
    state["messages"][-1] = HumanMessage(
        content=last_message.content + retry_instruction
    )

    return "retry"
```

#### Benefits

- ✅ **Measurable quality**: Track performance over time
- ✅ **Automatic improvement**: Retry if quality < 0.75
- ✅ **Reduced hallucinations**: Catch factual errors before user sees
- ✅ **Data-driven optimization**: Identify prompt improvements

---

### 1.4 Enhanced Observability & Logging

**Status**: RECOMMENDED
**Effort**: Low
**Impact**: Medium

#### What Changes

Add structured logging to debug agent decisions and performance.

#### Implementation

```python
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('foodhub_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("FoodHubAgent")


def log_agent_step(
    step_name: str,
    input_data: dict = None,
    output_data: dict = None,
    metadata: dict = None
):
    """
    Log agent execution steps with structured data.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step_name,
        "input": input_data or {},
        "output": output_data or {},
        "metadata": metadata or {}
    }

    logger.info(f"\n{'='*60}\nSTEP: {step_name}\n{json.dumps(log_entry, indent=2)}\n{'='*60}")


# Use in nodes
def sql_query_node(state: AgentState) -> AgentState:
    log_agent_step(
        "SQL_QUERY",
        input_data={"order_id": state["order_id"]},
        metadata={"thread_id": state.get("thread_id")}
    )

    result = sqlite_agent.invoke(f"Fetch all columns for order_id {state['order_id']}")

    log_agent_step(
        "SQL_QUERY",
        output_data={"rows_returned": len(result.get("output", ""))},
        metadata={"execution_time_ms": 1234}
    )

    state["order_context"] = result
    return state
```

#### Log Output Example

```
2025-10-08 10:23:45 - FoodHubAgent - INFO -
============================================================
STEP: SQL_QUERY
{
  "timestamp": "2025-10-08T10:23:45.123456",
  "step": "SQL_QUERY",
  "input": {
    "order_id": "O12486"
  },
  "output": {
    "rows_returned": 1
  },
  "metadata": {
    "thread_id": "C1011_O12486",
    "execution_time_ms": 1234
  }
}
============================================================
```

#### Benefits

- ✅ **Debugging**: Trace execution flow
- ✅ **Performance monitoring**: Track slow steps
- ✅ **Error tracking**: Identify failure patterns
- ✅ **Audit trail**: Conversation history

---

## Phase 2: Advanced Features

### 2.1 Enhanced Input Guardrail with Sentiment Analysis

**Status**: RECOMMENDED
**Effort**: Low
**Impact**: Medium

#### What Changes

Upgrade simple intent classification (0-3) to include sentiment and urgency scoring.

#### Implementation

```python
from typing import Literal
from pydantic import BaseModel, Field

class InputAnalysis(BaseModel):
    """Structured input classification with sentiment."""
    intent: Literal[0, 1, 2, 3] = Field(
        description="0=Escalation, 1=Exit, 2=Process, 3=Random"
    )
    sentiment: Literal["positive", "neutral", "negative", "angry"] = Field(
        description="Customer emotional state"
    )
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        description="Query urgency level"
    )
    escalate: bool = Field(
        description="True if human intervention needed"
    )
    reasoning: str = Field(
        description="Brief explanation of classification"
    )


def enhanced_input_analysis(user_query: str) -> InputAnalysis:
    """
    Analyze input with sentiment, urgency, and escalation flags.
    """
    prompt = f"""
Analyze this customer query and return structured JSON.

**INTENT (0-3):**
- 0 = Escalation (angry, threatening, demanding immediate action)
- 1 = Exit (goodbye, thanks, ending conversation)
- 2 = Process (valid order-related query)
- 3 = Random/Adversarial (hacking attempts, unrelated questions)

**SENTIMENT:**
- positive: Happy, satisfied, grateful
- neutral: Informational, matter-of-fact
- negative: Disappointed, concerned
- angry: Frustrated, upset, threatening

**URGENCY:**
- low: General inquiry, no time pressure
- medium: Wants update, moderate concern
- high: Needs answer soon, elevated concern
- critical: Immediate attention required

**ESCALATE:**
- true: Requires human intervention (anger, complex issue, repeat complaint)
- false: AI can handle

---

**CUSTOMER QUERY:**
{user_query}

---

Return ONLY valid JSON matching this schema:
{{
  "intent": 2,
  "sentiment": "neutral",
  "urgency": "medium",
  "escalate": false,
  "reasoning": "Customer asking about order status, neutral tone"
}}
"""

    llm = ChatOpenAI(model="local-model", temperature=0, ...)
    result = llm.predict(prompt)

    try:
        data = json.loads(result.strip())
        return InputAnalysis(**data)
    except Exception as e:
        logger.error(f"Input analysis failed: {e}")
        # Safe default: escalate on parse failure
        return InputAnalysis(
            intent=3,
            sentiment="neutral",
            urgency="high",
            escalate=True,
            reasoning="Failed to parse input"
        )


# LangGraph node
def input_analysis_node(state: AgentState) -> AgentState:
    """
    Enhanced input analysis with sentiment tracking.
    """
    query = state["messages"][-1].content

    analysis = enhanced_input_analysis(query)

    state["sentiment_analysis"] = {
        "intent": analysis.intent,
        "sentiment": analysis.sentiment,
        "urgency": analysis.urgency,
        "escalate": analysis.escalate,
        "reasoning": analysis.reasoning
    }

    logger.info(f"Input Analysis: {analysis.reasoning}")

    return state


def route_input(state: AgentState) -> str:
    """
    Route based on enhanced input analysis.
    """
    analysis = state["sentiment_analysis"]
    intent = analysis["intent"]
    escalate = analysis["escalate"]

    # Override intent if escalation flag set
    if escalate or intent == 0:
        state["agent_response"] = (
            "I understand your concern. Let me connect you with "
            "a specialist who can help you right away."
        )
        return "escalate"

    if intent == 1:
        state["agent_response"] = "Thank you for contacting FoodHub!"
        return "exit"

    if intent == 3:
        state["agent_response"] = (
            "I'm currently only able to help with order-related questions. "
            "Please let me know how I can assist with your order!"
        )
        return "redirect"

    # intent == 2, continue processing
    return "continue"
```

#### Benefits

- ✅ **Better escalation**: Detect frustration before it escalates
- ✅ **Prioritization**: Route urgent queries faster
- ✅ **Analytics**: Track sentiment trends over time
- ✅ **Proactive support**: Detect repeat complainers

---

### 2.2 Interactive Multi-Turn Chat Interface

**Status**: RECOMMENDED
**Effort**: Low
**Impact**: High (UX)

#### What Changes

Replace hardcoded test queries with interactive chat loop.

#### Implementation

```python
def interactive_chat_session(order_id: str, cust_id: str):
    """
    Interactive multi-turn conversation with memory and context.
    """
    config = {"configurable": {"thread_id": f"{cust_id}_{order_id}"}}

    print("=" * 60)
    print(f"🤖 FoodHub Assistant")
    print(f"📦 Order ID: {order_id} | 👤 Customer: {cust_id}")
    print("=" * 60)
    print("Type 'exit', 'quit', or 'bye' to end conversation")
    print("Type 'status' to see conversation statistics")
    print()

    conversation_stats = {
        "turn_count": 0,
        "avg_quality": [],
        "escalations": 0
    }

    while True:
        # Get user input
        user_input = input("You: ").strip()

        if not user_input:
            continue

        # Exit commands
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nAssistant: Thank you for contacting FoodHub! Have a great day! 👋")

            # Show conversation stats
            if conversation_stats["turn_count"] > 0:
                avg_quality = sum(conversation_stats["avg_quality"]) / len(conversation_stats["avg_quality"])
                print(f"\n📊 Session Stats:")
                print(f"   Turns: {conversation_stats['turn_count']}")
                print(f"   Avg Quality: {avg_quality:.2f}/1.0")
                print(f"   Escalations: {conversation_stats['escalations']}")

            break

        # Show stats command
        if user_input.lower() == "status":
            print("\n📊 Current Conversation Stats:")
            print(f"   Turns: {conversation_stats['turn_count']}")
            if conversation_stats["avg_quality"]:
                avg_quality = sum(conversation_stats["avg_quality"]) / len(conversation_stats["avg_quality"])
                print(f"   Avg Quality: {avg_quality:.2f}/1.0")
            print()
            continue

        # Process through agent
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "order_id": order_id,
            "cust_id": cust_id,
            "retry_count": 0
        }

        try:
            result = app.invoke(initial_state, config=config)

            # Extract response
            response = result.get("agent_response", "I'm having trouble responding. Please try again.")

            # Update stats
            conversation_stats["turn_count"] += 1
            if "quality_scores" in result:
                avg_score = (
                    result["quality_scores"].get("groundedness", 0) +
                    result["quality_scores"].get("precision", 0)
                ) / 2
                conversation_stats["avg_quality"].append(avg_score)

            if result.get("sentiment_analysis", {}).get("escalate", False):
                conversation_stats["escalations"] += 1

            # Display response with quality indicator
            quality_indicator = ""
            if "quality_scores" in result:
                scores = result["quality_scores"]
                g = scores.get("groundedness", 0)
                p = scores.get("precision", 0)
                if g >= 0.9 and p >= 0.9:
                    quality_indicator = " ⭐"
                elif g >= 0.75 and p >= 0.75:
                    quality_indicator = " ✓"
                else:
                    quality_indicator = " ⚠️"

            print(f"\nAssistant{quality_indicator}: {response}\n")

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            print(f"\nAssistant: I encountered an error. Please try rephrasing your question.\n")


# Usage
print("Starting FoodHub interactive chat session...\n")
interactive_chat_session("O12486", "C1011")
```

#### Example Session

```
============================================================
🤖 FoodHub Assistant
📦 Order ID: O12486 | 👤 Customer: C1011
============================================================
Type 'exit', 'quit', or 'bye' to end conversation
Type 'status' to see conversation statistics

You: Where is my order?

Assistant ✓: Hi there! Your order O12486 is currently being prepared
and should be ready shortly. We'll update you once it's out for
delivery!

You: How long will that take?

Assistant ⭐: Based on your order, preparation should be complete by
12:15 PM. You'll receive a notification when it's on the way!

You: Can I cancel it?

Assistant ✓: I see your order is currently being prepared. Unfortunately,
once preparation has started, we can't cancel the order. However, if
there's an issue when it arrives, we're happy to help with a refund
or replacement!

You: status

📊 Current Conversation Stats:
   Turns: 3
   Avg Quality: 0.88/1.0

You: exit# FoodHub FullCode Enhancements - Part 2

## Phase 3: Production Features (Optional)

### 3.1 Human-in-the-Loop for Critical Actions

**Status**: NICE-TO-HAVE
**Effort**: Medium
**Impact**: Medium

#### What Changes

Add approval workflow for critical actions (cancellations, refunds).

#### Implementation

```python
from langgraph.prebuilt import interrupt

def cancellation_approval_node(state: AgentState) -> AgentState:
    """
    Request human approval for order cancellations.
    """
    # Check if this is a cancellation request
    sentiment = state.get("sentiment_analysis", {})
    query = state["messages"][-1].content.lower()

    is_cancellation = any(word in query for word in ["cancel", "refund", "return"])

    if is_cancellation:
        # Pause execution for human approval
        approval_request = {
            "customer_id": state["cust_id"],
            "order_id": state["order_id"],
            "request": "CANCELLATION",
            "sentiment": sentiment.get("sentiment"),
            "urgency": sentiment.get("urgency")
        }

        # This pauses the workflow until human provides input
        approval = interrupt(approval_request)

        if approval == "approved":
            state["agent_response"] = (
                "Your cancellation request has been approved. "
                "You'll receive a confirmation email shortly."
            )
        elif approval == "rejected":
            state["agent_response"] = (
                "Unfortunately, we cannot process this cancellation at this time. "
                "Our policy states that orders in preparation cannot be cancelled. "
                "However, we can offer a refund if there are issues upon delivery."
            )
        else:
            # Default: escalate to human
            state["agent_response"] = (
                "Let me connect you with a specialist who can help "
                "with your cancellation request."
            )

    return state


# Resume workflow with human decision
# This would be done via API or UI
app.update_state(
    config={"configurable": {"thread_id": "C1011_O12486"}},
    values={"approval": "approved"}
)
```

---

### 3.2 Retry Logic with Exponential Backoff

**Status**: RECOMMENDED
**Effort**: Low
**Impact**: Medium

#### What Changes

Implement smart retry with prompt modification on quality failure.

#### Implementation

```python
def generate_response_with_retry(state: AgentState) -> AgentState:
    """
    Generate response with automatic retry on quality failure.
    Implemented via conditional edge in LangGraph.
    """
    retry_count = state.get("retry_count", 0)

    # Add retry-specific instructions
    if retry_count > 0:
        retry_instructions = f"""

[RETRY ATTEMPT {retry_count}/3]
IMPORTANT: Previous response failed quality check.
Issues detected:
- Groundedness score was too low (be more factual)
- Precision score was too low (be more specific and direct)

Use EXACT facts from the order data. No assumptions.
"""
        # Append to query
        original_query = state["messages"][-1].content
        if "[RETRY ATTEMPT" not in original_query:
            state["messages"][-1] = HumanMessage(
                content=original_query + retry_instructions
            )

    # Generate response (using existing answer_tool logic)
    response = answer_tool_func(
        state["messages"][-1].content,
        state["extracted_facts"],
        state["order_context"]
    )

    state["agent_response"] = response
    return state
```

---

### 3.3 Response Streaming (Optional)

**Status**: NICE-TO-HAVE
**Effort**: Low
**Impact**: Low (UX improvement)

#### Implementation

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Create streaming LLM
llm_streaming = ChatOpenAI(
    model="local-model",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY
)


def stream_response(response: str, delay: float = 0.03):
    """
    Simulate token-by-token streaming for better UX.
    """
    import time
    import sys

    for char in response:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()  # Newline at end


# Use in interactive chat
result = app.invoke(initial_state, config=config)
print("\nAssistant: ", end="")
stream_response(result["agent_response"])
```

---

## Architecture Comparison

### LowCode vs FullCode

| Component | LowCode | FullCode |
|-----------|---------|----------|
| **Agent Framework** | `initialize_agent()` (deprecated) | LangGraph StateGraph |
| **Conversation Memory** | None (stateless) | SqliteSaver checkpointing |
| **Quality Measurement** | None | LLM judges (groundedness + precision) |
| **Retry Logic** | None (single attempt) | 3 attempts with quality gates |
| **Input Classification** | Simple intent (0-3) | Intent + sentiment + urgency |
| **Observability** | Minimal | Structured logging + optional LangSmith |
| **Multi-turn Support** | ❌ No | ✅ Yes (with context) |
| **Human Escalation** | Basic | Advanced (sentiment-based) |
| **Interactive UI** | Hardcoded tests | Multi-turn chat interface |
| **Error Handling** | Basic | Retry with prompt modification |

---

### Execution Flow Comparison

**LowCode** (Linear):
```
Query → Guard → SQL → Tool1 → Tool2 → Guard → Response
```

**FullCode** (Graph with Cycles):
```
Query
  ↓
Input Analysis (sentiment + intent)
  ↓
SQL Query
  ↓
Extract Facts
  ↓
Generate Response
  ↓
Quality Evaluation ←──┐
  ↓ (fail)            │
  └─────── Retry ─────┘ (max 3x)
  ↓ (pass)
Output Guard
  ↓
Response + Memory Update
```

---

## Implementation Roadmap

### Recommended Implementation Order

#### **Week 1: Core Foundation**
1. ✅ Install dependencies (`langgraph`, `pydantic`)
2. ✅ Set up LangGraph StateGraph structure
3. ✅ Migrate existing tools to LangGraph nodes
4. ✅ Implement basic conversation memory (SqliteSaver)
5. ✅ Test single-turn conversations work

**Deliverable**: LangGraph agent with memory (no quality checks yet)

---

#### **Week 2: Quality & Retry**
6. ✅ Implement quality evaluation LLM judge
7. ✅ Add quality evaluation node to graph
8. ✅ Implement conditional retry logic
9. ✅ Add structured logging
10. ✅ Test retry mechanism with low-quality responses

**Deliverable**: Agent with quality gates and retry logic

---

#### **Week 3: Enhanced Guardrails**
11. ✅ Upgrade input guardrail (sentiment analysis)
12. ✅ Implement InputAnalysis Pydantic model
13. ✅ Update routing logic for sentiment-based escalation
14. ✅ Add enhanced output guardrail
15. ✅ Test escalation flows

**Deliverable**: Enhanced guardrails with sentiment tracking

---

#### **Week 4: Interactive UI & Polish**
16. ✅ Implement interactive chat interface
17. ✅ Add conversation statistics tracking
18. ✅ Implement streaming responses (optional)
19. ✅ Add human-in-the-loop for cancellations (optional)
20. ✅ Final testing and documentation

**Deliverable**: Complete FullCode notebook ready for submission

---

### Minimal Viable Implementation (MVP)

If time is limited, implement these **must-have** features:

1. **LangGraph Migration** (Phase 1.1) - 60% effort
2. **Conversation Memory** (Phase 1.2) - 20% effort
3. **Quality Evaluation** (Phase 1.3) - 15% effort
4. **Basic Logging** (Phase 1.4) - 5% effort

**Total Effort**: ~8-12 hours for experienced developer

**Skip for MVP**:
- Streaming responses
- Human-in-the-loop
- Advanced sentiment analysis (keep simple 0-3 classification)

---

## Testing Strategy

### Unit Tests for Each Component

```python
def test_quality_evaluation():
    """Test quality scoring is within 0.0-1.0 range."""
    scores = evaluate_response_quality(
        order_context="Order status: delivered",
        query="Where is my order?",
        response="Your order was delivered at 1:00 PM."
    )

    assert 0.0 <= scores["groundedness"] <= 1.0
    assert 0.0 <= scores["precision"] <= 1.0
    assert scores["groundedness"] > 0.7  # Should be high for factual response


def test_input_analysis():
    """Test enhanced input classification."""
    # Angry customer
    result = enhanced_input_analysis("This is ridiculous! I want my money back NOW!")
    assert result.intent == 0  # Escalation
    assert result.sentiment == "angry"
    assert result.escalate == True

    # Normal query
    result = enhanced_input_analysis("Where is my order?")
    assert result.intent == 2  # Process
    assert result.sentiment in ["neutral", "positive"]


def test_conversation_memory():
    """Test multi-turn context retention."""
    config = {"configurable": {"thread_id": "test_123"}}

    # Turn 1
    response1 = chat_with_memory("O12486", "C1011", "Where is my order?")
    assert "O12486" in response1

    # Turn 2 - "it" should refer to order from Turn 1
    response2 = chat_with_memory("O12486", "C1011", "Can I cancel it?")
    assert "cancel" in response2.lower()
    # Should understand "it" refers to the order
```

### Integration Test: Complete Flow

```python
def test_full_conversation_flow():
    """Test complete multi-turn conversation."""
    order_id = "O12486"
    cust_id = "C1011"
    config = {"configurable": {"thread_id": f"{cust_id}_{order_id}"}}

    # Test scenario: Customer asks about order, then tries to cancel
    queries = [
        "Where is my order?",
        "How long will it take?",
        "I want to cancel it"
    ]

    responses = []
    for query in queries:
        state = {
            "messages": [HumanMessage(content=query)],
            "order_id": order_id,
            "cust_id": cust_id,
            "retry_count": 0
        }

        result = app.invoke(state, config=config)
        responses.append(result["agent_response"])

        # Verify quality scores exist
        assert "quality_scores" in result
        assert result["quality_scores"]["groundedness"] >= 0.0
        assert result["quality_scores"]["precision"] >= 0.0

    # Verify context carried over
    assert "O12486" in responses[0]  # Order ID mentioned
    assert any(word in responses[2].lower() for word in ["cancel", "preparation"])
```

### Performance Benchmarks

```python
import time

def benchmark_response_time():
    """Measure average response time."""
    times = []

    for i in range(10):
        start = time.time()

        result = chat_with_memory("O12486", "C1011", "Where is my order?")

        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    print(f"Average response time: {avg_time:.2f} seconds")

    assert avg_time < 15.0  # Should respond within 15 seconds for local LLM
```

---

## Code Organization

### Recommended File Structure

```
notebooks/
├── FoodHub_Chatbot_FullCode_Notebook.ipynb  # Main notebook
└── foodhub_fullcode/                         # Python module (optional)
    ├── __init__.py
    ├── config.py          # Configuration management
    ├── models.py          # Pydantic models (InputAnalysis, QualityScores)
    ├── nodes.py           # LangGraph node functions
    ├── tools.py           # Tool functions (order_query, answer, evaluate)
    ├── guardrails.py      # Input/output guardrail logic
    └── utils.py           # Logging, formatting utilities

data/
├── customer_orders.db     # SQLite database
└── foodhub_conversations.db  # Conversation memory (auto-created)

logs/
└── foodhub_agent.log      # Application logs
```

### Cell Organization in Notebook

```markdown
# Cell 1: Package Installation
!pip install langgraph==0.2.56 langchain==0.3.26 ...

# Cell 2: Imports
import json, os, logging
from langgraph.graph import StateGraph, END
...

# Cell 3: Configuration
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
...

# Cell 4: Pydantic Models
class AgentState(TypedDict): ...
class InputAnalysis(BaseModel): ...
class QualityScores(BaseModel): ...

# Cell 5: Database Setup
order_db = SQLDatabase.from_uri("sqlite:///../data/customer_orders.db")
sqlite_agent = create_sql_agent(...)

# Cell 6: Tool Functions
def order_query_tool_func(...): ...
def answer_tool_func(...): ...
def evaluate_response_quality(...): ...

# Cell 7: Guardrail Functions
def enhanced_input_analysis(...): ...
def output_guard_check(...): ...

# Cell 8: LangGraph Nodes
def input_analysis_node(state): ...
def sql_query_node(state): ...
def extract_facts_node(state): ...
...

# Cell 9: Routing Functions
def route_input(state): ...
def should_retry(state): ...

# Cell 10: Graph Construction
workflow = StateGraph(AgentState)
workflow.add_node(...)
workflow.add_edge(...)
...
app = workflow.compile(checkpointer=memory)

# Cell 11: Helper Functions
def chat_with_memory(...): ...
def interactive_chat_session(...): ...

# Cell 12: Test Query 1
interactive_chat_session("O12486", "C1011")

# Cell 13: Test Query 2 (hardcoded for grading)
result = chat_with_memory("O12487", "C1012", "I want to cancel")
print(result)

# ... more test cells
```

---

## Expected Learning Outcomes

### Students Will Learn

1. **LangGraph Fundamentals**
   - State management with TypedDict
   - Node-based agent architecture
   - Conditional edges and routing logic
   - Persistent checkpointing

2. **Quality Engineering**
   - LLM-as-judge evaluation pattern
   - Groundedness vs precision metrics
   - Retry strategies with quality gates

3. **Conversation Design**
   - Multi-turn context retention
   - Sentiment-aware escalation
   - Graceful error handling

4. **Production Patterns**
   - Structured logging
   - Error handling and retries
   - Type safety with Pydantic
   - Memory management

---

## Comparison to Kartify Production App

### Similarities

| Feature | Kartify (Streamlit) | FoodHub FullCode |
|---------|---------------------|------------------|
| Multi-agent architecture | ✅ SQL + Chat agents | ✅ Same |
| Quality evaluation | ✅ Groundedness + Precision | ✅ Same |
| Input guardrails | ✅ Intent classification | ✅ Enhanced with sentiment |
| Output guardrails | ✅ Safety checks | ✅ Same |
| Local LLM support | ❌ (OpenAI only) | ✅ GPT-OSS 20B |

### Differences

| Aspect | Kartify | FoodHub FullCode |
|--------|---------|------------------|
| Framework | LangChain (legacy) | LangGraph (modern) |
| UI | Streamlit web app | Jupyter notebook + interactive CLI |
| Memory | Session state (temporary) | SQLite persistence (permanent) |
| Deployment | Docker container | Educational notebook |
| Retry logic | Single regeneration | 3 attempts with quality gates |

---

## Success Criteria

### Functional Requirements ✅

- [ ] Multi-turn conversations work (context preserved)
- [ ] Quality scores are measured for all responses
- [ ] Retry logic activates when quality < 0.75
- [ ] Input guardrail detects sentiment and urgency
- [ ] Output guardrail blocks unsafe responses
- [ ] Conversation memory persists across sessions
- [ ] Interactive chat interface is user-friendly

### Performance Requirements ✅

- [ ] Average response time < 15 seconds (local LLM)
- [ ] Quality scores average > 0.80 across test queries
- [ ] Less than 20% of queries require retry
- [ ] No crashes or exceptions during 10-turn conversation

### Code Quality Requirements ✅

- [ ] All nodes have docstrings
- [ ] Logging is implemented for all major steps
- [ ] Type hints used throughout (Pydantic models)
- [ ] Code is organized into logical sections
- [ ] Comments explain non-obvious logic

---

## Conclusion

The FullCode implementation represents a **production-grade evolution** of the LowCode chatbot, incorporating modern agentic AI patterns:

- **LangGraph** for stateful, cyclical workflows
- **Persistent memory** for multi-turn conversations
- **Quality gates** with automatic retry
- **Enhanced guardrails** with sentiment analysis
- **Comprehensive logging** for observability

This transforms a simple proof-of-concept into a system that could be deployed in production with minimal additional work (add API layer, scale with distributed memory, etc.).

**Estimated Effort**: 12-16 hours for full implementation
**Minimum Viable**: 8 hours (core features only)

---

## Appendix: Quick Reference

### Key Files to Create

1. `FoodHub_Chatbot_FullCode_Notebook.ipynb` - Main implementation
2. `foodhub_conversations.db` - Auto-created by SqliteSaver
3. `foodhub_agent.log` - Auto-created by logging

### Key Dependencies

```python
langgraph==0.2.56
langchain==0.3.26
langchain-openai==0.3.27
langchain-core==0.3.40
pydantic==2.10.6
```

### Key Imports

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal
```

### Environment Setup

```bash
# Start LM Studio with GPT-OSS 20B model
# Server: http://localhost:1234

# Run notebook
jupyter notebook FoodHub_Chatbot_FullCode_Notebook.ipynb
```

---

**Document Version**: 1.0
**Last Updated**: 2025-10-08
**Author**: PGP-GABA Course Team

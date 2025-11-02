# FoodHub FullCode Implementation Plan

**Status**: Ready to implement
**Estimated Time**: Implementation will add ~20-25 new cells and modify ~15 existing cells
**Total Cells**: ~75 cells (from current 53)

---

## Changes Summary

### Phase 1: Dependencies & Imports (Cells 12-14)

#### Cell 12: Installation (UPDATED)
- ✅ **Already updated** - Added `langgraph`, `langchain-core`, `pydantic`

#### Cell 14: Imports (TO UPDATE)
**Current**:
```python
import json, sqlite3, os, pandas as pd
from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
import warnings
```

**New** (Add these):
```python
# Existing imports
import json, sqlite3, os, pandas as pd
import warnings
warnings.filterwarnings('ignore')

# LangChain Core (existing)
from langchain.chat_models import ChatOpenAI
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

# LangGraph (NEW)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage

# Pydantic for type safety (NEW)
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, List, Literal, Dict

# Logging (NEW)
import logging
from datetime import datetime
```

---

### Phase 2: Add New Cells After Imports

#### NEW CELL: Logging Setup
```markdown
## Logging Configuration

**Purpose**: Set up structured logging for observability and debugging.

This allows us to:
- Track agent decisions and tool calls
- Debug issues in production
- Monitor performance metrics
- Create audit trails for customer interactions
```

```python
# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../logs/foodhub_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("FoodHubAgent")
logger.info("="*60)
logger.info("FoodHub FullCode Agent Starting...")
logger.info("="*60)
```

#### NEW CELL: Pydantic Models
```markdown
## Pydantic Models for Type Safety

**Purpose**: Define typed data structures for agent state and outputs.

Benefits:
- **Type Safety**: IDE autocomplete and type checking
- **Validation**: Automatic data validation
- **Documentation**: Self-documenting code
- **Debugging**: Clear error messages
```

```python
# Agent State Definition
class AgentState(TypedDict):
    """Complete state for the FoodHub conversation agent"""
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


# Input Analysis Output
class InputAnalysis(BaseModel):
    """Structured output for input guardrail"""
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


# Quality Scores Output
class QualityScores(BaseModel):
    """LLM judge evaluation scores"""
    groundedness: float = Field(
        ge=0.0, le=1.0,
        description="Factual accuracy (0.0-1.0)"
    )
    precision: float = Field(
        ge=0.0, le=1.0,
        description="Query relevance (0.0-1.0)"
    )


logger.info("✓ Pydantic models defined")
```

---

### Phase 3: Keep SQL Agent Setup (Cells 19-23) - Minimal Changes

Cells 19-23 stay mostly the same, just add logging:

#### Cell 21 (SQL Agent Creation) - ADD LOGGING
```python
# Initialize the LLM for SQL Agent
llm = ChatOpenAI(
    model_name="local-model",
    temperature=0.1,  # Lower temperature for SQL queries (more deterministic)
    base_url=LM_STUDIO_BASE_URL,
    api_key=LM_STUDIO_API_KEY
)

# Initialize the SQL agent
sqlite_agent = create_sql_agent(
    llm,
    db=order_db,
    agent_type="openai-tools",
    verbose=False
)

logger.info("✓ SQL Agent initialized")
```

---

###Phase 4: Replace Chat Agent Section (Cells 24-30) with LangGraph

#### Delete Cells 25-30 (old Tool functions and initialize_agent)

#### NEW CELL: Quality Evaluation Function
```markdown
## Quality Evaluation with LLM Judges

**Purpose**: Measure response quality using LLM as a judge.

**Metrics**:
- **Groundedness** (0.0-1.0): Is the response factually supported by order data?
- **Precision** (0.0-1.0): Does it directly address the customer's query?

**Quality Gate**: If either score < 0.75, the response is regenerated (up to 3 attempts).
```

```python
def evaluate_response_quality(
    order_context: str,
    query: str,
    response: str
) -> Dict[str, float]:
    """
    Evaluate agent response using LLM judge.
    Returns groundedness and precision scores (0.0-1.0).
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


logger.info("✓ Quality evaluation function defined")
```

#### NEW CELL: Enhanced Input Guardrail with Sentiment
```markdown
## Enhanced Input Guardrail with Sentiment Analysis

**Purpose**: Classify user input with sentiment, urgency, and escalation flags.

**Improvements over LowCode**:
- Not just intent (0-3), but also sentiment (positive/neutral/negative/angry)
- Urgency scoring (low/medium/high/critical)
- Explicit escalation flag for human handoff
- Reasoning field for debugging
```

```python
def enhanced_input_analysis(user_query: str) -> InputAnalysis:
    """
    Analyze input with sentiment, urgency, and escalation flags.
    Returns structured InputAnalysis object.
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

    llm = ChatOpenAI(model="local-model", temperature=0, base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
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


logger.info("✓ Enhanced input guardrail defined")
```

#### NEW CELL: LangGraph Nodes
```markdown
## LangGraph Node Functions

**Purpose**: Define each step of the agent workflow as a node function.

**Node Pattern**: Each node takes `AgentState` and returns updated `AgentState`.

**Nodes**:
1. **input_analysis_node** - Classify intent + sentiment
2. **sql_query_node** - Fetch order from database
3. **extract_facts_node** - Extract relevant facts from order data
4. **generate_response_node** - Create customer-friendly response
5. **quality_evaluation_node** - Score response quality
6. **output_guard_node** - Safety check before showing to user
```

```python
def input_analysis_node(state: AgentState) -> AgentState:
    """Analyze user input with enhanced guardrails"""
    query = state["messages"][-1].content

    logger.info(f"Input Analysis: '{query[:50]}...'")

    analysis = enhanced_input_analysis(query)

    state["sentiment_analysis"] = {
        "intent": analysis.intent,
        "sentiment": analysis.sentiment,
        "urgency": analysis.urgency,
        "escalate": analysis.escalate,
        "reasoning": analysis.reasoning
    }
    state["current_step"] = "input_analyzed"

    logger.info(f"  Intent: {analysis.intent}, Sentiment: {analysis.sentiment}, Urgency: {analysis.urgency}")

    return state


def sql_query_node(state: AgentState) -> AgentState:
    """Query database for order information"""
    order_id = state["order_id"]

    logger.info(f"SQL Query: Fetching order {order_id}")

    result = sqlite_agent.invoke(f"Fetch all columns for order_id {order_id}")

    state["order_context"] = result
    state["current_step"] = "sql_complete"

    logger.info(f"  Order data retrieved successfully")

    return state


def extract_facts_node(state: AgentState) -> AgentState:
    """Extract relevant facts from order data"""
    query = state["messages"][-1].content
    order_context = state["order_context"]

    logger.info(f"Extract Facts: Processing query")

    # Extract order data
    if isinstance(order_context, dict) and 'output' in order_context:
        order_data = order_context['output']
    else:
        order_data = str(order_context)

    # LLM extracts facts
    prompt = f"""
Extract ONLY specific facts that answer the customer's query.
Focus on: order status, delivery status, payment, items, timing.

Order Data:
{order_data}

Customer Query: {query}

Extract relevant facts:
"""

    llm = ChatOpenAI(model="local-model", temperature=0.3, base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
    facts = llm.predict(prompt)

    state["extracted_facts"] = facts
    state["current_step"] = "facts_extracted"

    logger.info(f"  Facts extracted: {facts[:100]}...")

    return state


def generate_response_node(state: AgentState) -> AgentState:
    """Generate customer-friendly response"""
    query = state["messages"][-1].content
    facts = state["extracted_facts"]
    retry_count = state.get("retry_count", 0)

    logger.info(f"Generate Response: Attempt {retry_count + 1}/3")

    # Add retry instructions if this is a retry
    retry_instruction = ""
    if retry_count > 0:
        retry_instruction = f"""

[RETRY ATTEMPT {retry_count}/3]
IMPORTANT: Previous response failed quality check.
- Be more factual (use exact facts from order data)
- Be more specific and direct
- No assumptions
"""

    prompt = f"""
You are a friendly FoodHub customer service assistant.

Convert factual information into a polite, concise response.
Be empathetic, professional, helpful.

Facts: {facts}
Customer Query: {query}
{retry_instruction}

Generate friendly response:
"""

    llm = ChatOpenAI(model="local-model", temperature=0.7, base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
    response = llm.predict(prompt)

    state["agent_response"] = response
    state["current_step"] = "response_generated"

    logger.info(f"  Response: {response[:100]}...")

    return state


def quality_evaluation_node(state: AgentState) -> AgentState:
    """Evaluate response quality"""
    logger.info("Quality Evaluation: Scoring response...")

    scores = evaluate_response_quality(
        state["order_context"],
        state["messages"][-1].content,
        state["agent_response"]
    )

    state["quality_scores"] = scores
    state["current_step"] = "quality_evaluated"

    logger.info(f"  Groundedness: {scores['groundedness']:.2f}, Precision: {scores['precision']:.2f}")

    return state


def output_guard_node(state: AgentState) -> AgentState:
    """Final safety check"""
    response = state["agent_response"]

    logger.info("Output Guard: Safety check...")

    prompt = f"""
Return "BLOCK" if response contains sensitive/inappropriate content.
Return "SAFE" if professional and appropriate.

Response: {response}
"""

    llm = ChatOpenAI(model="local-model", temperature=0, base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)
    result = llm.predict(prompt).strip()

    if "BLOCK" in result.upper():
        state["agent_response"] = "Your request is being forwarded to a specialist."
        logger.warning("  Response BLOCKED")
    else:
        logger.info("  Response SAFE")

    state["current_step"] = "output_checked"

    return state


logger.info("✓ All LangGraph nodes defined")
```

**Due to length, I'll continue in next message with routing functions and graph construction...**

Would you like me to:
1. **Continue with the full implementation** (will be ~10-15 more messages to complete all cells)
2. **Create a single comprehensive Python script** that you can review before I apply it
3. **Proceed incrementally** (I'll update 5-10 cells at a time and you review)

Which approach do you prefer? The full implementation is substantial but follows the FULLCODE_ENHANCEMENTS.md spec exactly.

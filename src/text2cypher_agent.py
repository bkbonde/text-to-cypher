#!/usr/bin/env python3
import json
import uuid
import sys
from typing import Optional 
from dotenv import load_dotenv

# LLM wrappers
from langchain_openai import ChatOpenAI

# Message history
from langchain_community.chat_message_histories import ChatMessageHistory

# Runnable with history (memory)
from langchain_core.runnables.history import RunnableWithMessageHistory

# Prompt templates
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


from src.utils import get_env_variable
from src.schema_loader import get_schema, get_schema_hints

from src.schema_cache import load_schema_once
from src.schema_compress import compress_schema
from src.schema_prompt import build_schema_prompt

load_dotenv()

# Single shared history store for all providers
_SHARED_HISTORY = ChatMessageHistory()
'''
SYSTEM_RULES = (
   "You are a Neo4j Cypher-generating assistant. You must strictly follow ALL rules below:\n\n"
   "iNSTRUTIONS:\n\n" \
   "CONSIDER VALID RELATIONSHIPS BEWEEN THE NODES AS PER THE PROVIDED SCHEMA.\n\n" \
   "USE ONLY THE PROVIDED SCHEMA TO GENERATE CYPHER QUERIES.\n\n" \
   "USE ONLY & ONLY THE SCHEMA , NODES, RELATIONSHIP BETWEEN THEM. DO NOT INVENT ANY NEW RELATIONSHIPS.HAVE BEEN PROVIDED TO YOU. DO NOT MAKE UP ANY LABELS, RELATIONSHIPS, OR PROPERTIES.\n\n" \
   
    "1. **Use ONLY node labels, relationship types, and property names that exist in the provided JSON schema.** "
    "Never invent, infer, or hallucinate labels, relationships, or properties.\n\n"

    "2. All text filtering on node properties MUST be case-insensitive. "
    "Always use: toLower(property) CONTAINS toLower('value') or toLower(property) = toLower('value').\n\n"

    "3. Every generated query MUST include: LIMIT 10 at the end.\n\n"

    "4. Cypher output MUST return a graph structure. "
    "Always return nodes and relationships (never return scalar values like names, counts, or strings).\n\n"

    "5. Every relationship MUST be assigned to a variable. "
    "If there is one relationship, use: r. "
    "If multiple relationships are used, name them r1, r2, r3, etc.\n\n"

    "6. Should use schema_hints to clarify ambiguous user requests. "

    "7. You may use multiple relationships to connect two nodes if a direct relationship does not exist, "
    "but ALL relationships must exist in the schema and must connect the correct node types.\n\n"

    "8. NEVER use variable-length paths like [*] or [*1..N].\n\n"

    "9. NEVER generate write operations such as CREATE, MERGE, DELETE, SET, or REMOVE. "
    "Only MATCH, OPTIONAL MATCH, WHERE, RETURN, and LIMIT are allowed.\n\n"

    "10. If the user references a label, relationship, or property that does not exist in the schema, "
    "map it to the closest valid schema element (exact match, substring, or best semantic match). "
    "Only ask for clarification if multiple matches are equally valid.\n\n"

    "11. Respond with Cypher ONLY. Do not include explanations, markdown, comments, or formatting.\n"
    "12. ALWAYS double-check that all labels, relationship types, and property names in your Cypher "
    

)
'''
SYSTEM_RULES = (
    "You are a Cypher-generating assistant. Follow these rules:\n"
    "1. Use *only* the node labels, relationship types and property names that appear in the JSON schema.\n"
    "2. If the user is revising a previous Cypher draft, update that draft; otherwise, generate a fresh query.\n"
    "3. Respond with **Cypher only** – no commentary or explanation.\n"
    "4. Return nodes and relationships only (omit scalar property values in RETURN).\n"
    "5. When the user mentions a label/relationship/property absent from the schema, first map it to the closest existing element (exact synonym, substring, or highest-similarity fuzzy match). Ask for clarification only if multiple matches are equally plausible, offering up to three suggestions.\n"
    "6. while generating cypher queries, use tolower() function for case insensitive matching of text properties.\n"
    "7. Always include 'LIMIT 10' at the end of each query.\n"
)


def make_llm(provider: str = "llama"):
    """Return a Chat instance for the specified provider."""
    if provider == "llama":
        return ChatOpenAI(
            base_url=get_env_variable("LLAMA_BASE_URL"),
            api_key="dummy",
            model=get_env_variable("LLAMA_MODEL"),
            temperature=0,
            request_timeout=20,   # 🔥 REQUIRED
            streaming=False
        )
    elif provider == "groq":
        return ChatOpenAI(
            base_url=get_env_variable("GROQ_BASE_URL"),
            model=get_env_variable("GROQ_MODEL"),
            api_key=get_env_variable("GROQ_API_KEY"),
            temperature=0
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")

class Text2CypherAgent:
    """Single‑LLM agent that remembers conversation context + schema."""

    def __init__(self, provider: str = "llama"):
        self.provider = provider
        self.schema_json = get_schema()
        self.schema_str = json.dumps(self.schema_json, indent=2)
        self.hints = get_schema_hints() 
        raw_schema = load_schema_once()
        schema_summary = compress_schema(raw_schema)
        schema_prompt = build_schema_prompt(schema_summary)

        system_prompt = SYSTEM_RULES + "\n" + schema_prompt

        
        # Build system prompt with schema and optional hints
        #whole_schema = self.schema_str.replace('{', '{{').replace('}', '}}')
        #system_prompt = SYSTEM_RULES + "\n### Schema\n" + whole_schema
        
        #if self.hints:
        #    hints_str = json.dumps(self.hints, indent=2).replace('{', '{{').replace('}', '}}')
        #    system_prompt += "\n\n### Schema Hints\n" + hints_str

        self.llm = make_llm(provider)
        self.session_id = "shared"  # All agents use same session for shared history

        # build prompt template with history placeholder
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}")
        ])

        chain_core = self.prompt | self.llm

        def get_history(session_id: str):
            return _SHARED_HISTORY

        self.chain = RunnableWithMessageHistory(
            chain_core,
            get_history,
            input_messages_key="user_input",
            history_messages_key="history",
        )

    def respond(self, user_text: str) -> str:
        print("before invoke")
        result = self.chain.invoke(
            {"user_input": user_text},
            config={"configurable": {"session_id": self.session_id}}
        )
        print("after invoke")
        return result.content.strip().strip("` ")

    def get_history(self) -> list[dict[str, str]]:
        """Return chat history as list of {role, content} dicts."""
        messages = []
        for m in _SHARED_HISTORY.messages:
            role = "assistant" if getattr(m, "type", "") == "ai" else "user"
            messages.append({"role": role, "content": m.content})
        return messages

    def clear_history(self) -> None:
        """Clear the shared history."""
        global _SHARED_HISTORY
        _SHARED_HISTORY = ChatMessageHistory()

if __name__ == "__main__":
    try:
        agent = Text2CypherAgent()
    except EnvironmentError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        while True:
            txt = input("You> ").strip()
            if not txt:
                continue
            print(agent.respond(txt) + "\n")
    except (KeyboardInterrupt, EOFError):
        print()
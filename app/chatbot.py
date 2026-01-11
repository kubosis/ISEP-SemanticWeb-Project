"""
file: _chatbot.py
brief: Simplified Academic Career Advisor Chatbot
"""
import dataclasses
import json
from typing import List, Dict, Any
import os

from openai import OpenAI
from loguru import logger
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

from app.utils import OPENAI_API_KEY, OPENAI_DEPLOYMENT, NEO4J_AUTH, NEO4J_URI


@dataclasses.dataclass
class ChatResponse:
    response: str
    tool_calls: List[Dict[str, Any]]

class CareerAdvisorChatbot:
    """
    Main class for the Academic Career Advisor.
    Manages conversation history and tool execution.
    """

    # 1. Define the Tools available to the LLM
    TOOLS_SCHEMA = [
        {
            "type": "function",
            "function": {
                "name": "search_job_market",
                "description": "Searches for job postings, required skills, and career trends in the Vector Database.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Job title or career description (e.g., 'Data Scientist', 'Marketing Manager')."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_academic_graph",
                "description": "Searches the academic graph for Courses, Skills, Topics, or Papers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entity_type": {
                            "type": "string",
                            "enum": ["Skill", "Topic", "Course", "Paper", "CourseResources"],
                            "description": "The type of lookup. Use 'CourseResources' to find papers for a specific course."
                        },
                        "search_term": {
                            "type": "string",
                            "description": "The name, title, or keyword to search for."
                        }
                    },
                    "required": ["entity_type", "search_term"]
                }
            }
        }

    ]

    SYSTEM_PROMPT = """
# Role
You are the "Academic Career Advisor," an intelligent assistant for university students. Your goal is to bridge the gap between a student's career aspirations and the academic curriculum. You help users decide what to study by analyzing real-world job market data and mapping it to specific university courses and research papers.

# Data Sources & Tools
You have access to two primary tools/databases. You must use them strictly according to their purposes:

1. **Job Market Intelligence (Vector Database - Qdrant):**
   - Contains real-world job postings and career descriptions.
   - **Use this when:** The user mentions a specific job title (e.g., "Data Scientist", "Product Manager") or describes a general career path.
   - **Goal:** Retrieve the required skills, responsibilities, and current market trends for that career.

2. **Academic Knowledge Graph (Graph Database - Neo4j):**
   - Contains the university curriculum, research papers, and standardized taxonomies.
   - **Nodes:** - `Course` (Properties: title, rawTopics, level, url)
     - `Paper` (Research papers related to courses)
     - `Skill` (ESCO standardized skills)
     - `Topic` (UNESCO academic subjects)
     - `Keyword` (Normalized search terms)
   - **Key Relationships:**
     - `(:Course)-[:TEACHES_SKILL]->(:Skill)`: Crucial for linking jobs to courses.
     - `(:Paper)-[:RELATED_TO_COURSE]->(:Course)`: For deep-dive research recommendations.
     - `(:Course)-[:HAS_TOPIC]->(:Topic)`: For broad subject grouping.
   - **Use this when:** You need to find specific courses that teach a skill, find papers on a topic, or explore the academic details of a subject.

# operational Workflow
When a user asks "How do I become a [Job Title]?":
1. **Analyze Demand:** Query the **Vector DB** to find job postings for [Job Title]. Extract the most frequent/important skills mentioned in those postings.
2. **Map to Curriculum:** query the **Graph DB** to find `Course` nodes that connect to those specific `Skill` nodes (via `[:TEACHES_SKILL]`) or `Keyword` nodes.
3. **Enrich:** Look for `Paper` nodes connected to those courses if the user wants to deepen their knowledge (`[:RELATED_TO_COURSE]`).
4. **Synthesize:** Present the answer by linking the job requirement directly to the specific course.

# Response Guidelines
- **Be Evidence-Based:** Do not invent courses. Only recommend courses found in the Graph DB.
- **Explain "Why":** Don't just list courses. Say: *"You should take [Course Name] because it teaches [Skill], which is frequently required in [Job Title] listings."*
- **Use Standard Terminology:** When discussing skills, refer to them by their ESCO labels. When discussing subjects, use UNESCO topics.
- **Cite Sources:** If you recommend a paper, provide the title and year. If you recommend a course, provide the course title and level.
- **Tone:** Encouraging, professional, academic, and practical.

# Constraints
- If you cannot find a direct link between a job skill and a course, acknowledge this gap honestly; do not hallucinate a connection.
- If the user asks about a topic outside of the school's curriculum, inform them that the current database does not cover that area.
    """

    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = OPENAI_DEPLOYMENT

        # Initialize Conversation History with System Prompt
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT.strip()}]

        # Tool Dispatch Table
        self.available_functions = {
            "search_job_market": self._query_qdrant,
            "search_academic_graph": self._query_neo4j,
        }

        self.driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

        # Initialize Qdrant with error recovery
        self.qdrant_client = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))
        self.embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        self.qdrant_collection = "job_market"

        # Attempt to verify/recover collection
        self._initialize_qdrant_collection()

        # Initialize LangChain wrapper
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.qdrant_collection,
            embedding=self.embedding_model
        )

        logger.success("CareerAdvisorChatbot initialized.")

    def _initialize_qdrant_collection(self):
        """
        Verifies Qdrant collection health and recreates if corrupted.
        """
        try:
            if self.qdrant_client.collection_exists(self.qdrant_collection):
                # Try to get collection info to verify it's not corrupted
                info = self.qdrant_client.get_collection(self.qdrant_collection)
                logger.info(
                    f"Qdrant collection '{self.qdrant_collection}' exists: "
                    f"{info.points_count} points, "
                    f"vector size: {info.config.params.vectors.size}"
                )

                # Verify vector size matches
                if info.config.params.vectors.size != 1536:
                    logger.warning(
                        f"Vector size mismatch! Expected 1536, got {info.config.params.vectors.size}. "
                        "Recreating collection..."
                    )
                    self._recreate_collection()

            else:
                logger.warning(f"Collection '{self.qdrant_collection}' does not exist. Creating...")
                self._recreate_collection()

        except Exception as e:
            logger.error(f"Qdrant collection check failed: {e}")
            logger.warning("Collection may be corrupted. Attempting to recreate...")
            self._recreate_collection()

    def _recreate_collection(self):
        """
        Safely recreates the Qdrant collection.
        """
        from qdrant_client.models import Distance, VectorParams

        try:
            # Try to delete if exists (may fail if corrupted, that's OK)
            try:
                self.qdrant_client.delete_collection(self.qdrant_collection)
                logger.info(f"Deleted existing collection: {self.qdrant_collection}")
            except Exception as e:
                logger.warning(f"Could not delete collection (may not exist): {e}")

            # Create fresh collection
            self.qdrant_client.create_collection(
                collection_name=self.qdrant_collection,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
            )
            logger.success(f"Created fresh collection: {self.qdrant_collection}")
            logger.warning("⚠️  Collection is now empty. You need to run ingestion to populate it.")

        except Exception as e:
            logger.error(f"Failed to recreate collection: {e}")
            raise RuntimeError(
                "Cannot initialize Qdrant. Please check Qdrant service and storage."
            )

    def _query_qdrant(self, query: str) -> str:
        """
        Searches the Qdrant vector database with improved error handling.
        """
        logger.info(f"Tool Action: Querying Qdrant for '{query}'")

        try:
            # 1. Verify collection exists and is accessible
            if not self.qdrant_client.collection_exists(self.qdrant_collection):
                logger.warning("Collection doesn't exist")
                return json.dumps({
                    "error": "Job database not initialized. Please contact administrator."
                })

            # 2. Check collection health and size
            try:
                collection_info = self.qdrant_client.get_collection(self.qdrant_collection)
                vector_count = collection_info.points_count

                if vector_count == 0:
                    logger.warning("Collection exists but is empty")
                    return json.dumps({
                        "message": "Job market database is empty. No postings available yet."
                    })

                logger.debug(f"Collection has {vector_count} vectors")

            except Exception as e:
                logger.error(f"Failed to get collection info: {e}")
                return json.dumps({
                    "error": "Job database may be corrupted. Please contact administrator.",
                    "details": "Collection health check failed"
                })

            # 3. Perform Search with timeout protection
            results = self.vector_store.similarity_search_with_score(
                query=query,
                k=min(4, vector_count)  # Don't request more than available
            )

            if not results:
                return json.dumps({
                    "message": "No relevant job postings found for your query."
                })

            # 4. Format Results
            found_jobs = []
            for doc, score in results:
                found_jobs.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "score": round(float(score), 3),
                    "content": doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                })

            logger.info(f"Found {len(found_jobs)} relevant job postings")
            return json.dumps(found_jobs)

        except Exception as e:
            logger.error(f"Qdrant Search Error: {e}")
            return json.dumps({
                "error": "Failed to search job market database.",
                "details": str(e),
                "suggestion": "The database may need to be reinitialized."
            })

    def _query_neo4j(self, entity_type: str, search_term: str) -> str:
        """
        Executes a Fulltext Search query to handle fuzzy matching and relevance.
        """
        logger.info(f"Tool Action: Querying Neo4j Index for {entity_type}: '{search_term}'")

        # We append '~' to the term to enable fuzzy matching (handling typos)
        fuzzy_term = f"{search_term}~"

        queries = {
            "Skill": """
                // Use the Fulltext Index to find the best matching Skill nodes
                CALL db.index.fulltext.queryNodes("skillNames", $term) YIELD node, score
                WITH node AS s, score
                // Get the courses connected to these skills
                MATCH (c:Course)-[:TEACHES_SKILL]->(s)
                RETURN 
                    s.label AS MatchedSkill, 
                    score AS Relevance,
                    c.courseTitle AS Course, 
                    c.level AS Level, 
                    c.url AS URL
                ORDER BY score DESC
                LIMIT 5
            """,

            "Topic": """
                // Search Topics OR Keywords
                CALL db.index.fulltext.queryNodes("topicNames", $term) YIELD node, score
                WITH node AS t, score
                MATCH (t)<-[:HAS_TOPIC]-(c:Course)
                RETURN 
                    t.name AS Topic,
                    c.courseTitle AS Course,
                    c.url AS URL,
                    score
                ORDER BY score DESC
                LIMIT 5
            """,

            "Course": """
                CALL db.index.fulltext.queryNodes("courseNames", $term) YIELD node, score
                RETURN 
                    node.courseTitle AS Title, 
                    node.rawTopics AS Topics, 
                    node.level AS Level, 
                    score
                ORDER BY score DESC
                LIMIT 1
            """
        }

        cypher_query = queries.get(entity_type)
        if not cypher_query:
            return json.dumps({"error": f"Invalid entity_type: {entity_type}"})

        try:
            with self.driver.session() as session:
                # Pass the fuzzy term to the query
                result = session.run(cypher_query, term=fuzzy_term)
                records = [record.data() for record in result]

                if not records:
                    # If index fails, try the Keyword fallback (synonyms)
                    if entity_type == "Skill":
                        return self._fallback_keyword_search(search_term)
                    return json.dumps({"message": f"No records found for '{search_term}'."})

                return json.dumps(records, default=str)

        except Exception as e:
            logger.error(f"Neo4j Query Error: {e}")
            return json.dumps({"error": "Database query failed", "details": str(e)})

    def _fallback_keyword_search(self, search_term: str) -> str:
        """
        Fallback: Search the Keyword index if the specific entity index failed.
        """
        logger.info(f"Fallback: Searching Keywords for '{search_term}'")
        query = """
            CALL db.index.fulltext.queryNodes("keywordNames", $term) YIELD node, score
            MATCH (c:Course)-[:MENTIONS]->(node)
            RETURN 
                node.name AS MatchedKeyword,
                c.courseTitle AS Course,
                score
            ORDER BY score DESC LIMIT 3
        """
        with self.driver.session() as session:
            result = session.run(query, term=f"{search_term}~")
            records = [record.data() for record in result]
            if not records:
                return json.dumps({"message": f"No records found."})
            return json.dumps(records, default=str)

    # --- Main Logic ---

    def chat(self, user_query: str) -> ChatResponse:
        """
        Handles the full chat interaction:
        User -> LLM -> (Optional Tool Calls) -> LLM -> Response
        Supports iterative tool calling until LLM provides final answer.
        """
        # 1. Append User Message
        self.messages.append({"role": "user", "content": user_query})

        tool_calls_data = []
        max_iterations = 5  # Prevent infinite loops

        for iteration in range(max_iterations):
            # 2. Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=self.TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.1,
            )
            response_msg = response.choices[0].message

            # 3. Check if LLM wants to use tools
            if response_msg.tool_calls:
                logger.info(f"LLM requested tool execution (iteration {iteration + 1})...")
                self.messages.append(response_msg)  # Add assistant's tool request to history

                # Execute each tool call
                for tool_call in response_msg.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    # Execute the internal function
                    function_to_call = self.available_functions[function_name]

                    if function_name == "search_job_market":
                        function_response = function_to_call(query=function_args.get("query"))
                    elif function_name == "search_academic_graph":
                        function_response = function_to_call(
                            entity_type=function_args.get("entity_type"),
                            search_term=function_args.get("search_term")
                        )
                    else:
                        function_response = json.dumps({"error": f"Unknown function: {function_name}"})

                    tool_calls_data.append({
                        "tool": function_name,
                        "args": function_args,
                        "result": function_response
                    })

                    # Append Tool Result to conversation
                    self.messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    })

                # Continue loop to let LLM process tool results
                continue

            else:
                # No more tools needed - LLM provided final answer
                final_content = response_msg.content

                # Append final response to history
                self.messages.append({"role": "assistant", "content": final_content})

                return ChatResponse(response=final_content, tool_calls=tool_calls_data)

        # If we hit max iterations, return what we have
        logger.warning(f"Hit max iterations ({max_iterations}) - forcing completion")
        error_msg = "I apologize, but I needed to make too many tool calls. Please try rephrasing your question."
        self.messages.append({"role": "assistant", "content": error_msg})
        return ChatResponse(response=error_msg, tool_calls=tool_calls_data)

    def reset(self):
        """Clears conversation history."""
        self.messages = [{"role": "system", "content": self.SYSTEM_PROMPT.strip()}]
        logger.info("Context cleared.")

# Usage Example
if __name__ == "__main__":
    bot = CareerAdvisorChatbot()
    ans = bot.chat("How do I become a Data Scientist?")
    print(f"Bot: {ans.response}")
    print(f"Tools Used: {ans.tool_calls}")
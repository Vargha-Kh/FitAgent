import os

from phi.knowledge.langchain import LangChainKnowledgeBase
from langchain_openai import OpenAIEmbeddings
from utils import get_vectorstores
from phi.tools.duckduckgo import DuckDuckGo
from phi.agent import Agent, AgentMemory
from phi.model.openai import OpenAIChat
from phi.memory.db.sqlite import SqliteMemoryDb
from phi.storage.agent.sqlite import SqlAgentStorage
from phi.knowledge.text import TextKnowledgeBase
# from phi.vectordb.weaviate import Weaviate
from phi.playground import Playground, serve_playground_app


class FitFusion:
    """
    A utility class for building an AI-driven diet and workout planning system
    using LangChain-based knowledge and an agent architecture.
    """

    def __init__(
            self,
            llm_model: str,
            data_path: str,
            data_types=None,
            vectorstore_name: str = "weaviate",
            embeddings_model: str = "openai",
            create_db: bool = True
    ):
        """
        Initialize FitFusion with model and database settings.

        Args:
            llm_model (str): LLM model identifier (e.g. "gpt-4o").
            data_path (str): Directory containing text files or other data.
            data_types (list): Types of files to load as knowledge.
            vectorstore_name (str): Name of the vector store to use (e.g. "weaviate").
            embeddings_model (str): Name of the embeddings model to use (e.g. "openai").
        """
        if data_types is None:
            data_types = ["txt"]
        self.model_type = llm_model
        self.data_path = data_path
        self.data_types = data_types
        self.vectorstore_name = vectorstore_name
        self.embeddings_model = embeddings_model

        self.database_collection_name = "RAG"
        self.chunk_size = 5000
        self.create_db = create_db
        self.db_file = "./data.db"

        # Additional placeholders
        self.agent = None
        self.loader = None
        self.results = None
        self.text_splitter = None
        self.model = None
        self.temperature = 0.1
        self.chain = None
        self.result = None
        self.chat_history = []

    def _select_embeddings_model(self):
        """
        Pick an embeddings model based on the embeddings_model attribute.

        Returns:
            OpenAIEmbeddings or a different embeddings class.
        """
        # Example placeholder for a custom OllamaEmbeddings:
        # if self.embeddings_model == "ollama":
        #     return OllamaEmbeddings(model=self.model_type)
        return OpenAIEmbeddings(model="text-embedding-3-small")

    def create_knowledge_base(self):
        """
        Create a knowledge base using the specified vector store and embeddings.

        Returns:
            LangChainKnowledgeBase: An object with retrieval and storage functionality.
        """
        vector_store = get_vectorstores(
            vectorstore_name=self.vectorstore_name,
            data_path=self.data_path,
            data_types=self.data_types,
            embedding_function=self._select_embeddings_model(),
            collection_name=self.database_collection_name,
            chunk_size=self.chunk_size,
            create_db=self.create_db,
        )
        knowledge_base = LangChainKnowledgeBase(
            retriever=vector_store.as_retriever(),
            vectorstore=vector_store,
        )
        return knowledge_base

    # # Integrated weaviate vectordb
    # def create_knowledge_base(self):
    #     """
    #     Create a knowledge base using the specified vector store and embeddings.
    #
    #     Returns:
    #         LangChainKnowledgeBase: An object with retrieval and storage functionality.
    #     """
    #     knowledge_base = TextKnowledgeBase(
    #         path=self.data_path,
    #         vector_db=Weaviate(index_name=self.database_collection_name),
    #     )
    #
    #     return knowledge_base

    def agent_init(self) -> Agent:
        """
        Initialize and link multiple agents to handle queries.

        Returns:
            Agent: A team-based agent that coordinates sub-agents (diet_agent, web_searcher).
        """
        knowledge_base = self.create_knowledge_base()
        # Comment out after first run
        # knowledge_base.load(recreate=True)

        # Sub-agent that focuses on building diet & workout plans.
        diet_agent = Agent(
            model=OpenAIChat(id=self.model_type),
            name="diet and workout coach",
            description="Expert diet and workout planner.",
            instructions=[
                "You are tasked with creating a **step-by-step plan** to develop a **7-day diet and workout routine** using the retrieved data from the knowledge base and the user's provided information. ",
                "Use the user’s personal details (age, gender, height, weight, goals, dietary preferences, workout preferences, etc.) to ensure the plan is well-tailored. ",
                "Your plan should include daily meals (breakfast, lunch, dinner, snacks) and a corresponding workout schedule for each day with specifying exercise details and plan, accounting for rest days as necessary. ",
                "In each step, reference the relevant knowledge or insights from the knowledge base—do not assume information that isn’t verified or retrieved. ",
                "Ensure your final step provides a **complete 7-day plan**, incorporating meal details, macro guidelines, workout details, and any additional health/wellness considerations. ",
                "Make sure do not skip any days plan",
                "Do not add superfluous steps—only steps that contribute to creating and presenting the plan. ",
                "The result of the **final step** is your final answer."
            ],
            knowledge=knowledge_base,
            search_knowledge=True,
            add_references_to_prompt=True,
            reasoning=False,
            show_tool_calls=True,
            markdown=True,
        )

        # Sub-agent that can perform external web searches.
        web_searcher = Agent(
            name="Web Searcher",
            model=OpenAIChat(id=self.model_type),
            role="Searches the web for relevant diet/workout information.",
            tools=[DuckDuckGo()],
            add_datetime_to_instructions=True,
        )

        # Combine sub-agents into a single team agent.
        self.agent = Agent(
            name="Diet and Workout Planner Team",
            team=[diet_agent, web_searcher],
            instructions=[
                "First, retrieve any relevant information from the knowledge base that addresses the user's query or domain of interest.",
                "Then, ask the article reader to review the retrieved documents for detailed insights. Provide direct references or links to the content so it can be examined.",
                "Next, if additional information is needed, request the web searcher to gather supplementary data from external sources. Provide those links to the article reader as well, ensuring thorough research.",
                "Finally, synthesize all the gathered information into a clear, step-by-step 7-day diet and workout plan tailored to the user's details and goals and Make sure do not skip any days plan."
            ],
            show_tool_calls=True,
            markdown=True,
            read_chat_history=True,
            memory=AgentMemory(
                db=SqliteMemoryDb(table_name="agent_memory", db_file=self.db_file),
                create_user_memories=True,
                create_session_summary=True
            ),
            storage=SqlAgentStorage(
                table_name="agent_sessions",
                db_file=self.db_file,
            )
        )
        return self.agent

    def query_inferences(self, query_input: str) -> None:
        """
        Send a query to the agent to receive a response.

        Args:
            query_input (str): A textual query or request.
        """
        self.agent.print_response(query_input, stream=True)


dummy_user_info = f"""
        Generate Diet plan with full instruction and scheduling based on the information:
        ### Personal Information
        Age: 25
        Gender: Male
        Height: 180 cm
        Weight: 70 kg

        ### Goals
        Primary Goal: Muscle Gain
        Target Weight: 78 kg
        Timeframe: 3 months

        ### Activity Levels
        Current Physical Activity: Moderately active (fitness enthusiast)

        ### Medical and Health Information
        Existing Medical Conditions: None
        Food Allergies: None

        ### Dietary Preferences
        Diet Type: Omnivore
        Meal Frequency Preferences: 3 meals + 2 snacks

        ### Workout Preferences
        Preferred Workout Types: Cardio, Strength training
        Current Fitness Level: Beginner
        Workout Frequency: 3 days/week
        Workout Duration: ~45 min

        ### Lifestyle and Habits
        Sleep Patterns: ~8 hours
        Stress Levels: low
        Hydration Habits: ~3L water/day

        ### Behavioral Insights
        Motivators: General Health, Appearance
        Barriers: Time constraints (office job)

        ### Feedback and Customization
        Adjustability: Willing to adjust plan each month
        Feedback Loop: Weekly weigh-ins and monthly measurements
        """

# ------------------------------------------------------------------------------
# Create main app
# ------------------------------------------------------------------------------

# 1. Instantiate the FitFusion class
_fit_fusion_instance = FitFusion(
    llm_model="gpt-4o-mini",
    data_path="./diet_data",
    data_types=["txt"],
    vectorstore_name="weaviate",
    embeddings_model="openai",
    create_db=True
)

# 2. Initialize the agent
agent = _fit_fusion_instance.agent_init()

# 3. Generate the Playground FastAPI `app`
app = Playground(agents=[agent]).get_app()


def main():
    # For launching Streamlit interface
    # serve_playground_app("fit_fusion:app", reload=True)
    # For running in terminal
    _fit_fusion_instance.query_inferences(dummy_user_info)


if __name__ == "__main__":
    main()

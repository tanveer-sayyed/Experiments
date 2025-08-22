from langchain.schema import Document
   
def create_mythical_creatures_db():
    documents = [
        Document(
            page_content="""Phoenix - A magnificent bird that cyclically regenerates by bursting into flames upon death and being reborn from the ashes. 
            Known for its healing tears and ability to be reborn, making it a symbol of renewal and resurrection.""",
            metadata={"creature_type": "Bird", "origin": "Greek", "magical_ability": "Rebirth"}
        ),
        Document(
            page_content="""Dragon - Majestic, serpentine creatures with the ability to breathe fire. 
            Often depicted as guardians of great treasures and possessing ancient wisdom.""",
            metadata={"creature_type": "Reptile", "origin": "Global", "magical_ability": "Fire Breathing"}
        ),
        Document(
            page_content="""Unicorn - A horse-like creature with a single, spiraling horn on its forehead. 
            Their horns are said to have the power to heal sickness and purify water.""",
            metadata={"creature_type": "Equine", "origin": "European", "magical_ability": "Healing"}
        ),
        Document(
            page_content="""Kitsune - Japanese fox spirits with intelligence, long life, and magical abilities. 
            They can shapeshift into human form and are known for their trickery and wisdom.""",
            metadata={"creature_type": "Canine", "origin": "Japanese", "magical_ability": "Shapeshifting"}
        ),
        Document(
            page_content="""Kraken - A legendary sea monster of enormous size, said to appear off the coasts of Norway and Greenland. 
            Known to attack ships and drag them to the ocean depths.""",
            metadata={"creature_type": "Cephalopod", "origin": "Norse", "magical_ability": "Whirlpool Creation"}
        )
    ]
    return documents


from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.manager import BaseCallbackManager
from langchain import hub
from langchain_ollama import OllamaLLM

from asyncLogsAndMetrics import monitor
from callbackHandler import CustomAsyncCallbacks

async def main(question):
    thread_id = "retriever"
    logger, populateMetrics, metric = await monitor(thread_id)
    metric.user.thread_id = thread_id
    prompt = hub.pull("rlm/rag-prompt")
    callback_manager = BaseCallbackManager(
        handlers=[
            CustomAsyncCallbacks(
                logger=logger,
                populateMetrics=populateMetrics
                )
            ]
        )
    llm = OllamaLLM(
        model="mistral:7b",
        base_url="http://localhost:11435",
        # callback_manager=callback_manager,
        verbose=True
        )
    embedding = OllamaEmbeddings(
        model="mistral:7b",
        base_url="http://localhost:11435" # default is 11434
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=25,
        )
    splitted = splitter.split_documents(create_mythical_creatures_db())
    vector_db = FAISS.from_documents(splitted, embedding=embedding)
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k":3}
        )
    joinDocs = lambda docs: "\n".join([d.page_content for d in docs])
    rag_chain = (
        {"context":retriever | joinDocs, "question":RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    return await rag_chain.ainvoke(question, config={"callbacks": callback_manager})

# await main(question="What creatures are from Japanese mythology?")

# queries = [
#     "Find creatures that can heal",
#     "Show me fire-related mythical beings",
#     "What creatures are from Japanese mythology?",
#     "Find creatures that live in water",
#     "List of "
# ]
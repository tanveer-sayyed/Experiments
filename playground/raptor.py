from langchain_community.document_loaders import WikipediaLoader
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer

query = "Hayao Miyazaki"
raw_documents = WikipediaLoader(query=query).load()
doc_text = [d.page_content for d in raw_documents]

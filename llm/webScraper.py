from langchain_community.document_loaders import UnstructuredURLLoader
page_content = UnstructuredURLLoader(
    urls=["https://en.wikipedia.org/wiki/Security_incidents_involving_Donald_Trump"]
    ).load()[0].to_json()['kwargs']['page_content']




from langchain_community.document_loaders import UnstructuredURLLoader
page_content = UnstructuredURLLoader(
    urls=["https://www.nseindia.com/option-chain?symbolCode=-10003&symbol=NIFTY&symbol=NIFTY&instrument=OPTIDX&date=-&segmentLink=17&segmentLink=17"]
    ).load()[0].to_json()['kwargs']['page_content']
print(page_content)



from json import JSONEncoder
from langchain_core.runnables.graph import UUID

class UUIDEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID): return str(obj)
        return JSONEncoder.default(self, obj)

from uuid import UUID
from json import JSONEncoder

class aobject(object):
    """https://stackoverflow.com/questions/33128325/how-to-set-class-attribute-with-await-in-init"""
    async def __new__(cls, *a, **kw):
        instance = super().__new__(cls)
        await instance.__init__(*a, **kw)
        return instance
    async def __init__(self):
        pass
    
class UUIDEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, UUID): return str(obj)
        return JSONEncoder.default(self, obj)

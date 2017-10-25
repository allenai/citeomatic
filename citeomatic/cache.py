import json

STRING_ENCODING = 'utf-8'


class Cache(object):
    def lookup(self, namespace: str, key: str):
        raise NotImplementedError("Please use subclass")

    def put(self, namespace: str, key: str, json_str: str):
        raise NotImplementedError("Please use subclass")


class LocalCache(Cache):
    def __init__(self):
        self._dict = {}

    def create_hash_key(self, namespace: str, key: str):
        return "%s/%s" % (namespace, key)

    def lookup(self, namespace: str, key: str):
        hash_key = self.create_hash_key(namespace, key)
        return self._dict.get(hash_key, None)

    def put(self, namespace: str, key: str, json_str: str):
        hash_key = self.create_hash_key(namespace, key)
        self._dict[hash_key] = json.loads(json_str)

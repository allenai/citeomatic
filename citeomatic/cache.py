import json

import boto3
import botocore.exceptions

STRING_ENCODING = 'utf-8'


class Cache(object):
    def lookup(self, namespace: str, key: str):
        raise NotImplementedError("Please use subclass")

    def put(self, namespace: str, key: str, json_str: str):
        raise NotImplementedError("Please use subclass")


class S3Cache(Cache):
    def __init__(self, bucket):
        self.s3 = boto3.resource('s3')
        self.bucket = bucket

    # Creates an AWS S3 Key for storage
    def create_s3_key(self, namespace: str, key: str):
        return "%s/%s.json" % (namespace, key)

    def file_exists_in_s3(self, key: str):
        try:
            self.s3.Object(self.bucket, key).load()
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] == "404":
                return False
            else:
                raise
        else:
            return True

    def find_file_in_s3(self, key: str):
        if self.file_exists_in_s3(key):
            return self.s3.Object(self.bucket, key
                                 ).get()['Body'].read().decode(STRING_ENCODING)
        return None

    def lookup(self, namespace: str, key: str):
        s3_key = self.create_s3_key(namespace, key)
        s3_file = self.find_file_in_s3(s3_key)
        if s3_file:
            return json.loads(s3_file)
        else:
            return None

    def put(self, namespace: str, key: str, json_str: str):
        s3_key = self.create_s3_key(namespace, key)
        if json:
            self.s3.Object(self.bucket,
                           s3_key).put(Body=json_str.encode(STRING_ENCODING))


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

import pymongo
import json


def connect(host, port, user, passw):
    client = None
    if user and passw:
        uri = "mongodb://{}:{}@{}:{}/test".format(user, passw, host, port)
        client = pymongo.MongoClient(uri)
    else:
        client = pymongo.MongoClient(host, port)
    if client is None:
        print("can not connect to mongo at host: [{}], port [{}], username: [{}], password: [{}]".format(host, port,
                                                                                                         user, passw))
        return None
    try:
        dbnames = client.database_names()
    except pymongo.errors.ServerSelectionTimeoutError:
        print("can not get database from mongo at host: {}, port {}, connection timed out".format(host,port))
        return None
    if "reporting_db" not in dbnames:
        print("no database for results found")
        return None
    return client.reporting_db


def log2json(log):
    s=[]
    with open(log) as f:
        for line in f:
            x = line.replace("'", '"')
            s.append(json.loads(x))
    return s


def read_config(config):
    with open(config) as f:
        return json.load(f)

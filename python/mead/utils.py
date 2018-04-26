import json
import os
import re
import hashlib
import shutil
import requests
from clint.textui import progress
import magic


def extract_gzip(file_loc):
    import gzip
    import shutil
    temp_file = "{}.1".format(file_loc)
    with gzip.open(file_loc, 'rb') as f_in:
        with open(temp_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    shutil.move(temp_file, file_loc)
    return file_loc


def extract_zip(file_loc):
    import zipfile
    temp_file = "{}.1".format(file_loc)
    with zipfile.ZipFile(file_loc, "r") as zip_ref:
        zip_ref.extractall(temp_file)
    if os.path.isdir(temp_file):
        if len(os.listdir(temp_file)) > 1:
            raise RuntimeError("the directory extracted from the downloaded link contains more than one files, should contain only one")
        temp_file_path = os.path.join(temp_file, os.listdir(temp_file)[0])
    shutil.move(temp_file_path, file_loc)
    return file_loc



zipd = {'application/gzip': extract_gzip, 'application/zip': extract_zip}


def file_dloader(url, cachedir):
    r = requests.get(url, stream=True)
    path_to_save = "/tmp/data.dload-{}".format(os.getpid())
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    try:
        print("downloading {}".format(url))
        with open(path_to_save, 'wb') as f:
            total_length = int(r.headers.get('content-length'))
            for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
                if chunk:
                    f.write(chunk)
                    f.flush()
        sha1 = hashlib.sha1(open(path_to_save, 'rb').read()).hexdigest()
        if mime_type(path_to_save) in zipd:
            print("uncompressing data")
            path_to_save = zipd[mime_type(path_to_save)](path_to_save)
        path_to_save_sha1 = "{}/{}".format(cachedir, sha1)
        shutil.move(path_to_save, path_to_save_sha1)
        print("downloaded data saved in {}".format(path_to_save_sha1))
    except:  # this is too broad as an exception clause, should be changed probably? but there are too many exceptions to handle separately
        raise RuntimeError("failed to download data from [url]: {} [to]: {}".format(url, path_to_save))
    return path_to_save_sha1


def validate_url(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex,url) is not None


def index_by_label(dataset_file):
    with open(dataset_file) as f:
        datasets_list = json.load(f)
        datasets = dict((x["label"], x) for x in datasets_list)
        return datasets


def download(file_loc, cache_reset=False):
    if os.path.exists(file_loc):
        return file_loc
    elif validate_url(file_loc):  # is it a web URL? check if exists in cache
        url = file_loc
        try:
            dcaches = json.load(open("config/datasets-embeddings-cache.json"))
        except IOError:
            dcaches = {}
        if url in dcaches and not cache_reset:
            return dcaches[url]
        else:  # dload the file in the cache, update the json
            cache_dir = json.load(open("config.json"))["datacache"]
            dload_file = file_dloader(url, cache_dir)
            if dload_file:
                dcaches.update({url: dload_file})
                json.dump(dcaches, open("config/datasets-embeddings-cache.json", "w"))
                return dload_file
            else:
                raise RuntimeError("the file {} is not in cache and can not be downloaded".format(file_loc))

    else:
        raise RuntimeError("the file {} is not in cache and can not be downloaded".format(file_loc))


def mime_type(loc):
    return magic.Magic(mime=True).from_file(loc)

def extract_gzip_onefile(file_loc):
    import gzip
    import shutil
    temp_file = "{}.1".format(file_loc)
    with gzip.open(file_loc, 'rb') as f_in:
        with open(temp_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    shutil.move(temp_file, file_loc)
    return file_loc


def extract_zip_onefile(file_loc):
    import zipfile
    import shutil
    import os
    temp_file = "{}.1".format(file_loc)
    with zipfile.ZipFile(file_loc, "r") as zip_ref:
        zip_ref.extractall(temp_file)
    if os.path.isdir(temp_file):
        if len(os.listdir(temp_file)) > 1:
            raise RuntimeError("the directory extracted from the downloaded link contains more than one files, should contain only one")
        temp_file_path = os.path.join(temp_file, os.listdir(temp_file)[0])
    shutil.move(temp_file_path, file_loc)
    return file_loc


def extract_gzip_dir(file_loc):
    import tarfile
    import os
    temp_file = "{}.1".format(file_loc)
    with tarfile.open(file_loc, "r") as tar_ref:
        tar_ref.extractall(temp_file)
    if len(os.listdir(temp_file)) != 1:
        raise RuntimeError("tar extraction unsuccessful")
    return os.path.join(temp_file, os.listdir(temp_file)[0])


def extract_zip_dir(file_loc):
    import zipfile
    temp_file = "{}.1".format(file_loc)
    with zipfile.ZipFile(file_loc, "r") as zip_ref:
        zip_ref.extractall(temp_file)
    return temp_file

def mime_type(loc):
    import magic
    return magic.Magic(mime=True).from_file(loc)


def extractor(filepath, cache_dir, extractor_func):
    import hashlib
    import shutil
    import os
    sha1 = hashlib.sha1(open(filepath, 'rb').read()).hexdigest()
    path_to_save = sha1 if extractor_func is None else extractor_func(filepath)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    path_to_save_sha1 = "{}/{}".format(cache_dir, sha1)
    shutil.move(path_to_save, path_to_save_sha1)
    print("downloaded data saved in {}".format(path_to_save_sha1))
    return path_to_save_sha1


def web_downloader(url):
    import requests
    import shutil
    from clint.textui import progress
    import os
    r = requests.get(url, stream=True)
    path_to_save = "/tmp/data.dload-{}".format(os.getpid())
    print(path_to_save)
    try:
        print("downloading {}".format(url))
        total_length = None
        with open(path_to_save, 'wb') as f:
            try:
                total_length = int(r.headers.get('content-length'))
            except TypeError:
                print("download size can not be calculated")
            if total_length is not None:
                for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1):
                    if chunk:
                        f.write(chunk)
                        f.flush()
            else:
                shutil.copyfileobj(r.raw, f)  # only this worked! https://stackoverflow.com/a/18043472/8321467

    except:  # this is too broad but there are too many exceptions to handle separately
        raise RuntimeError("failed to download data from [url]: {} [to]: {}".format(url, path_to_save))
    return path_to_save


def validate_url(url):
    import re
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None


class Downloader(object):
    def __init(self):
        pass

    def download(self):
        pass


class SingleFileDownloader(Downloader):
    def __init__(self, datasetfile):
        super(Downloader, self).__init__()
        self.datasetfile = datasetfile

    def download(self):
        import json
        import os
        file_loc = self.datasetfile
        if os.path.exists(file_loc):
            return file_loc
        elif validate_url(file_loc):  # is it a web URL? check if exists in cache
            url = file_loc
            try:
                dcaches = json.load(open("config/datasets-embeddings-cache.json"))
            except IOError:
                dcaches = {}
            if url in dcaches and not self.cache_reset:
                return dcaches[url]
            else:  # download the file in the cache, update the json
                cache_dir = json.load(open("config.json"))["datacache"]
                zipd = {'application/gzip': extract_gzip_onefile, 'application/zip': extract_zip_onefile}
                temp_file = web_downloader(url)
                dload_file = extractor(filepath=temp_file, cache_dir=cache_dir, extractor_func=zipd.get(mime_type(temp_file), None))
                dcaches.update({url: dload_file})
                json.dump(dcaches, open("config/datasets-embeddings-cache.json", "w"), indent=True)
                return dload_file
        else: # try to see if we can validate it in some way
            raise RuntimeError("the file {} is not in cache and can not be downloaded".format(file_loc))


class DirDownloader(Downloader):  # will return locations to train, dev, test files

    def __init__(self, datasetdesc):
        super(Downloader, self).__init__()
        self.datasetdesc = datasetdesc

    def download(self):
        import os
        import json
        dload_bundle = self.datasetdesc.get("download", None)

        if dload_bundle is not None:  # download a zip/tar/tar.gz directory, look for train, dev test files inside that.
            try:
                download_caches = json.load(open("config/datasets-embeddings-cache.json"))
            except IOError:
                download_caches = {}
            if dload_bundle in download_caches and not self.cache_reset:
                download_dir = download_caches[dload_bundle]
                return {k: os.path.join(download_dir, self.datasetdesc[k]) for k in self.datasetdesc if k.endswith("_file")}
            else:  # try to download the bundle and unzip
                url = dload_bundle
                if not validate_url(url):
                    raise RuntimeError("can not download from the given url")
                else:
                    cache_dir = json.load(open("config.json"))["datacache"]
                    temp_file = web_downloader(url)
                    zipd = {'application/gzip': extract_gzip_dir, 'application/zip': extract_zip_dir}
                    download_dir = extractor(filepath=temp_file, cache_dir=cache_dir, extractor_func=zipd.get(mime_type(temp_file), None))
                    download_caches.update({url: download_dir})
                    json.dump(download_caches, open("config/datasets-embeddings-cache.json", "w"), indent=True)
                    return {k: os.path.join(download_dir, self.datasetdesc[k]) for k in self.datasetdesc if k.endswith("_file")}
        else:  # we have download links to every file or they exist
            return {k: self.datasetdesc[k] if os.path.exists(self.datasetdesc[k]) else SingleFileDownloader(None, self.datasetdesc[k]).download()
                    for k in self.datasetdesc if k.endswith("_file")}








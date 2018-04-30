def get_file(file_name):
    import shutil
    import os
    if os.path.exists(file_name):
        shutil.rmtree(file_name)
    return file_name


def extract_gzip(file_loc):
    import gzip
    import shutil
    temp_file = get_file("{}.1".format(file_loc))
    with gzip.open(file_loc, 'rb') as f_in:
        with open(temp_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if mime_type(temp_file) == "application/x-tar":
        return extract_tar(temp_file)
    else:
        shutil.move(temp_file, file_loc)
        return file_loc


def extract_tar(file_loc):
    import tarfile
    import os
    temp_file = get_file("{}.1".format(file_loc))
    with tarfile.open(file_loc, "r") as tar_ref:
        tar_ref.extractall(temp_file)
    if len(os.listdir(temp_file)) != 1:
        raise RuntimeError("tar extraction unsuccessful")
    return os.path.join(temp_file, os.listdir(temp_file)[0])


def extract_zip(file_loc):
    import zipfile
    temp_file = get_file("{}.1".format(file_loc))
    with zipfile.ZipFile(file_loc, "r") as zip_ref:
        zip_ref.extractall(temp_file)
    return temp_file


def mime_type(loc):
    import magic
    return magic.Magic(mime=True).from_file(loc)


def get_cache_dir():
    import json
    try:
        data_cache = json.load(open("config.json")).get("datacache", "~/.bl-dataset-embeddings/")
    except IOError:
        data_cache = "~/.bl-dataset-embeddings/"

    return data_cache


def extractor(filepath, cache_dir, extractor_func):
    import hashlib
    import shutil
    import os
    sha1 = hashlib.sha1(open(filepath, 'rb').read()).hexdigest()
    print("extracting file..")
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
    from baseline.progress import create_progress_bar
    import os
    r = requests.get(url, stream=True)
    path_to_save = "/tmp/data.dload-{}".format(os.getpid())
    try:
        print("downloading {}".format(url))
        total_length = None
        with open(path_to_save, 'wb') as f:
            try:
                total_length = int(r.headers.get('content-length'))
            except TypeError:
                print("download size can not be calculated")
            if total_length is not None:
                chunk_size=1024
                pg = create_progress_bar(total_length/chunk_size)
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pg.update()
                        f.flush()
                pg.done()
            else:
                shutil.copyfileobj(r.raw, f)

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
    def __init__(self, data_download_cache, cache_ignore):
        super(Downloader, self).__init__()
        self.cache_ignore = cache_ignore
        self.data_download_cache = data_download_cache
        pass

    def download(self):
        pass


class SingleFileDownloader(Downloader):
    def __init__(self, dataset_file, data_download_cache, cache_ignore=False):
        super(SingleFileDownloader, self).__init__(data_download_cache, cache_ignore)
        self.dataset_file = dataset_file
        self.data_download_cache = data_download_cache

    def download(self):
        import json
        import os
        file_loc = self.dataset_file
        if os.path.exists(file_loc):
            return file_loc
        elif validate_url(file_loc):  # is it a web URL? check if exists in cache
            url = file_loc
            try:
                dcaches = json.load(open("config/datasets-embeddings-cache.json"))
            except IOError:
                dcaches = {}
            if url in dcaches and not self.cache_ignore:
                print("file for {} found in cache, not downloading".format(url))
                return dcaches[url]
            else:  # download the file in the cache, update the json
                cache_dir = self.data_download_cache
                print("using {} as data/embeddings cache".format(cache_dir))
                zipd = {'application/gzip': extract_gzip, 'application/zip': extract_zip}
                temp_file = web_downloader(url)
                dload_file = extractor(filepath=temp_file, cache_dir=cache_dir, extractor_func=zipd.get(mime_type(temp_file), None))
                dcaches.update({url: dload_file})
                json.dump(dcaches, open("config/datasets-embeddings-cache.json", "w"), indent=True)
                return dload_file
        else:  # try to see if we can validate it in some way
            raise RuntimeError("the file {} is not in cache and can not be downloaded".format(file_loc))


class DataDownloader(Downloader):
    def __init__(self, dataset_desc, data_download_cache, enc_dec=False, cache_ignore=False):
        super(DataDownloader, self).__init__(data_download_cache, cache_ignore)
        self.dataset_desc = dataset_desc
        self.data_download_cache = data_download_cache
        self.enc_dec = enc_dec

    def download(self):
        import os
        import json
        dload_bundle = self.dataset_desc.get("download", None)
        if dload_bundle is not None:  # download a zip/tar/tar.gz directory, look for train, dev test files inside that.
            try:
                download_caches = json.load(open("config/datasets-embeddings-cache.json"))
            except IOError:
                download_caches = {}
            if dload_bundle in download_caches and not self.cache_ignore:
                download_dir = download_caches[dload_bundle]
                print("files for {} found in cache, not downloading".format(dload_bundle))
                return {k: os.path.join(download_dir, self.dataset_desc[k]) for k in self.dataset_desc if k.endswith("_file")}
            else:  # try to download the bundle and unzip
                url = dload_bundle
                if not validate_url(url):
                    raise RuntimeError("can not download from the given url")
                else:
                    cache_dir = self.data_download_cache
                    temp_file = web_downloader(url)
                    zipd = {'application/gzip': extract_gzip, 'application/zip': extract_zip}
                    download_dir = extractor(filepath=temp_file, cache_dir=cache_dir, extractor_func=zipd.get(mime_type(temp_file), None))
                    download_caches.update({url: download_dir})
                    json.dump(download_caches, open("config/datasets-embeddings-cache.json", "w"), indent=True)
                    return {k: os.path.join(download_dir, self.dataset_desc[k]) for k in self.dataset_desc if k.endswith("_file")}
        else:  # we have download links to every file or they exist
            if not self.enc_dec:
                return {k: SingleFileDownloader(self.dataset_desc[k], self.data_download_cache).download() for k in self.dataset_desc if k.endswith("_file")}
            else:
                return {k: self.dataset_desc[k] for k in self.dataset_desc if k.endswith("_file")} # these files can not be downloaded because there's a post processing on them.


class EmbeddingDownloader(Downloader):
    def __init__(self, embedding_file, embedding_dsz, data_download_cache, cache_ignore=False):
        super(EmbeddingDownloader, self).__init__(data_download_cache, cache_ignore)
        self.embedding_file = embedding_file
        self.embedding_key = embedding_dsz
        self.data_download_cache = data_download_cache

    @staticmethod
    def _get_embedding_file(loc, key):
        import os
        if os.path.isfile(loc):
            print("embedding file location: {}".format(loc))
            return loc
        else:  # This is a directory, return the actual file
            files = [x for x in os.listdir(loc) if str(key) in x]
            if len(files) == 0:
                raise RuntimeError("No embedding file found for the given key [{}]".format(key))
            elif len(files) > 1:
                print("multiple embedding files found for the given key [{}], choosing {}".format(key, files[0]))
            embed_file_loc = os.path.join(loc, files[0])
            print("embedding file location: {}".format(embed_file_loc))
            return embed_file_loc

    def download(self):
        import json
        import os
        if os.path.exists(self.embedding_file):
            print("embedding file location: {}".format(self.embedding_file))
            return self.embedding_file
        try:
            download_caches = json.load(open("config/datasets-embeddings-cache.json"))
        except IOError:
            download_caches = {}
        if self.embedding_file in download_caches and not self.cache_ignore:
            download_loc = download_caches[self.embedding_file]
            print("files for {} found in cache, not downloading".format(self.embedding_file))
            return self._get_embedding_file(download_loc, self.embedding_key)
        else:  # try to download the bundle and unzip
            url = self.embedding_file
            if not validate_url(url):
                raise RuntimeError("can not download from the given url")
            else:
                cache_dir = self.data_download_cache
                temp_file = web_downloader(url)
                zipd = {'application/gzip': extract_gzip, 'application/zip': extract_zip}
                download_loc = extractor(filepath=temp_file, cache_dir=cache_dir, extractor_func=zipd.get(mime_type(temp_file), None))
                download_caches.update({url: download_loc})
                json.dump(download_caches, open("config/datasets-embeddings-cache.json", "w"), indent=True)
                return self._get_embedding_file(download_loc, self.embedding_key)

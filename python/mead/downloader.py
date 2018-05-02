import gzip
import tarfile
import zipfile
import magic
import hashlib
import shutil
import requests
import os
import re
import json


LOG = lambda x: print(x)
DATA_CACHE_CONF = "data-cache.json"


def delete_old_copy(file_name):
    if os.path.exists(file_name):
        if os.path.isfile(file_name):
            os.remove(file_name)
        else:
            shutil.rmtree(file_name)
    return file_name


def extract_gzip(file_loc):
    temp_file = delete_old_copy("{}.1".format(file_loc))
    with gzip.open(file_loc, 'rb') as f_in:
        with open(temp_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if mime_type(temp_file) == "application/x-tar":
        return extract_tar(temp_file)
    else:
        shutil.move(temp_file, file_loc)
        return file_loc


def extract_tar(file_loc):
    temp_file = delete_old_copy("{}.1".format(file_loc))
    with tarfile.open(file_loc, "r") as tar_ref:
        tar_ref.extractall(temp_file)
    if len(os.listdir(temp_file)) != 1:
        raise RuntimeError("tar extraction unsuccessful")
    return os.path.join(temp_file, os.listdir(temp_file)[0])


def extract_zip(file_loc):
    temp_file = delete_old_copy("{}.1".format(file_loc))
    with zipfile.ZipFile(file_loc, "r") as zip_ref:
        zip_ref.extractall(temp_file)
    return temp_file


def mime_type(loc):
    return magic.Magic(mime=True).from_file(loc)


def extractor(filepath, cache_dir, extractor_func):
    sha1 = hashlib.sha1(open(filepath, 'rb').read()).hexdigest()
    LOG("extracting file..")
    path_to_save = filepath if extractor_func is None else extractor_func(filepath)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    path_to_save_sha1 = os.path.join(cache_dir, sha1)
    shutil.move(path_to_save, path_to_save_sha1)
    LOG("downloaded data saved in {}".format(path_to_save_sha1))
    return path_to_save_sha1


def web_downloader(url):
    from baseline.progress import create_progress_bar
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError("The file can not be downloaded")
    path_to_save = "/tmp/data.dload-{}".format(os.getpid())
    try:
        LOG("downloading {}".format(url))
        with open(path_to_save, 'wb') as f:
            total_length = r.headers.get('content-length', None)
            if total_length is not None:
                total_length = int(total_length)
                chunk_size = 8*1024  # experimented with a couple, this seemed fastest.
                pg = create_progress_bar(total_length/chunk_size)
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pg.update()
                        f.flush()
                pg.done()
            else:
                raise RuntimeWarning("Total length can not be calculated, aborting download")

    except:  # this is too broad but there are too many exceptions to handle separately
        raise RuntimeError("failed to download data from [url]: {} [to]: {}".format(url, path_to_save))
    return path_to_save


def validate_url(url):
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return re.match(regex, url) is not None


def update_cache(key, data_download_cache):
    dcache = read_json(os.path.join(data_download_cache, DATA_CACHE_CONF))
    if key not in dcache:
        return
    del dcache[key]
    write_to_json(dcache, os.path.join(data_download_cache, DATA_CACHE_CONF))


def check_sanity_file(file_loc, data_dcache=None, key=None):
    if os.path.exists(file_loc) and os.path.isfile(file_loc) and not mime_type(file_loc) == "text/html":
        # dropbox doesn't give 404 in case the file does not exist, produces an HTML. The actual files are never HTMLs.
        return True
    else:
        delete_old_copy(file_loc)
        if key is not None:  # cache file validation
            update_cache(key, data_dcache)
        return False


def check_sanity_dir(dir_loc, dataset_desc, data_dcache=None, ignore_file_check=False, key=None):
    if not os.path.exists(dir_loc) or not os.path.isdir(dir_loc):
        update_cache(key, data_dcache)
        return False
    if ignore_file_check:  # for enc_dec tasks there's no direct downloads
        return True
    files = [os.path.join(dir_loc, dataset_desc[k]) for k in dataset_desc if k.endswith("_file")]
    for file in files:
        if not check_sanity_file(file, key, data_dcache):
            return False
    return True


def read_json(filepath):
    if not os.path.exists(filepath):
        return {}
    with open(filepath) as f:
        j_con = json.load(f)
    return j_con


def write_to_json(content, filepath):
    with open(filepath, "w") as f:
        json.dump(content, f, indent=True)


class Downloader(object):
    def __init__(self, data_download_cache, cache_ignore):
        super(Downloader, self).__init__()
        self.cache_ignore = cache_ignore
        self.data_download_cache = data_download_cache

    def download(self):
        pass


class SingleFileDownloader(Downloader):
    def __init__(self, dataset_file, data_download_cache, cache_ignore=False):
        super(SingleFileDownloader, self).__init__(data_download_cache, cache_ignore)
        self.dataset_file = dataset_file
        self.data_download_cache = data_download_cache
        self.zipd = {'application/gzip': extract_gzip, 'application/zip': extract_zip}

    def download(self):
        file_loc = self.dataset_file
        if check_sanity_file(file_loc):
            return file_loc
        elif validate_url(file_loc):  # is it a web URL? check if exists in cache
            url = file_loc
            dcache_path = os.path.join(self.data_download_cache, DATA_CACHE_CONF)
            dcache = read_json(dcache_path)
            if url in dcache and check_sanity_file(dcache[url], self.data_download_cache, url) and not self.cache_ignore:
                LOG("file for {} found in cache, not downloading".format(url))
                return dcache[url]
            else:  # download the file in the cache, update the json
                cache_dir = self.data_download_cache
                LOG("using {} as data/embeddings cache".format(cache_dir))
                temp_file = web_downloader(url)
                dload_file = extractor(filepath=temp_file, cache_dir=cache_dir,
                                       extractor_func=self.zipd.get(mime_type(temp_file), None))
                dcache.update({url: dload_file})
                write_to_json(dcache, os.path.join(self.data_download_cache, DATA_CACHE_CONF))
                return dload_file
        else:
            raise RuntimeError("the file {} is not in cache and can not be downloaded".format(file_loc))


class DataDownloader(Downloader):
    def __init__(self, dataset_desc, data_download_cache, enc_dec=False, cache_ignore=False):
        super(DataDownloader, self).__init__(data_download_cache, cache_ignore)
        self.dataset_desc = dataset_desc
        self.data_download_cache = data_download_cache
        self.enc_dec = enc_dec
        self.zipd = {'application/gzip': extract_gzip, 'application/zip': extract_zip}

    def download(self):
        dload_bundle = self.dataset_desc.get("download", None)
        if dload_bundle is not None:  # download a zip/tar/tar.gz directory, look for train, dev test files inside that.
            dcache_path = os.path.join(self.data_download_cache, DATA_CACHE_CONF)
            dcache = read_json(dcache_path)
            if dload_bundle in dcache and \
                    check_sanity_dir(dcache[dload_bundle], self.dataset_desc, self.data_download_cache, self.enc_dec, dload_bundle)\
                    and not self.cache_ignore:
                download_dir = dcache[dload_bundle]
                LOG("files for {} found in cache, not downloading".format(dload_bundle))
                return {k: os.path.join(download_dir, self.dataset_desc[k]) for k in self.dataset_desc
                        if k.endswith("_file")}
            else:  # try to download the bundle and unzip
                if not validate_url(dload_bundle):
                    raise RuntimeError("can not download from the given url")
                else:
                    cache_dir = self.data_download_cache
                    temp_file = web_downloader(dload_bundle)

                    download_dir = extractor(filepath=temp_file, cache_dir=cache_dir,
                                             extractor_func=self.zipd.get(mime_type(temp_file), None))
                    if "sha1" in self.dataset_desc:
                        if os.path.split(download_dir)[-1] != self.dataset_desc["sha1"]:
                            raise RuntimeError("The sha1 of the downloaded file does not match with the provided one")
                    dcache.update({dload_bundle: download_dir})
                    write_to_json(dcache, os.path.join(self.data_download_cache, DATA_CACHE_CONF))
                    return {k: os.path.join(download_dir, self.dataset_desc[k]) for k in self.dataset_desc
                            if k.endswith("_file")}
        else:  # we have download links to every file or they exist
            if not self.enc_dec:
                    return {k: SingleFileDownloader(self.dataset_desc[k], self.data_download_cache).download()
                        for k in self.dataset_desc if k.endswith("_file")}
            else:
                return {k: self.dataset_desc[k] for k in self.dataset_desc if k.endswith("_file")}
                # these files can not be downloaded because there's a post processing on them.


class EmbeddingDownloader(Downloader):
    def __init__(self, embedding_file, embedding_dsz, embedding_sha1, data_download_cache, cache_ignore=False):
        super(EmbeddingDownloader, self).__init__(data_download_cache, cache_ignore)
        self.embedding_file = embedding_file
        self.embedding_key = embedding_dsz
        self.data_download_cache = data_download_cache
        self.sha1 = embedding_sha1
        self.zipd = {'application/gzip': extract_gzip, 'application/zip': extract_zip}

    def _get_embedding_file(self, loc, key, dload_url):
        if os.path.isfile(loc):
            if check_sanity_file(loc):
                LOG("embedding file location: {}".format(loc))
                return loc
            else:
                return EmbeddingDownloader(self.embedding_file, self.embedding_key, self.data_download_cache).download()
        else:  # This is a directory, return the actual file
            files = [x for x in os.listdir(loc) if str(key) in x]
            if len(files) == 0:
                raise RuntimeError("No embedding file found for the given key [{}]".format(key))
            elif len(files) > 1:
                LOG("multiple embedding files found for the given key [{}], choosing {}".format(key, files[0]))
            embed_file_loc = os.path.join(loc, files[0])
            if check_sanity_file(embed_file_loc, self.data_download_cache, dload_url):
                LOG("embedding file location: {}".format(embed_file_loc))
                return embed_file_loc
            else:
                return EmbeddingDownloader(self.embedding_file, self.embedding_key, self.data_download_cache).download()

    def download(self):
        if check_sanity_file(self.embedding_file):
            LOG("embedding file location: {}".format(self.embedding_file))
            return self.embedding_file
        dcache_path = os.path.join(self.data_download_cache, DATA_CACHE_CONF)
        dcache = read_json(dcache_path)
        if self.embedding_file in dcache and not self.cache_ignore:
            download_loc = dcache[self.embedding_file]
            LOG("files for {} found in cache".format(self.embedding_file))
            return self._get_embedding_file(download_loc, self.embedding_key, self.embedding_file)
        else:  # try to download the bundle and unzip
            url = self.embedding_file
            if not validate_url(url):
                raise RuntimeError("can not download from the given url")
            else:
                cache_dir = self.data_download_cache
                temp_file = web_downloader(url)
                download_loc = extractor(filepath=temp_file, cache_dir=cache_dir,
                                         extractor_func=self.zipd.get(mime_type(temp_file), None))
                if self.sha1 is not None:
                    if os.path.split(download_loc) != self.sha1:
                        raise RuntimeError("The sha1 of the downloaded file does not match with the provided one")
                dcache.update({url: download_loc})
                write_to_json(dcache, os.path.join(self.data_download_cache, DATA_CACHE_CONF))
                return self._get_embedding_file(download_loc, self.embedding_key, self.embedding_file)

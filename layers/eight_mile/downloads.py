import gzip
import hashlib
import logging
import os
import shutil
import zipfile
from six.moves.urllib.request import urlretrieve
from eight_mile.utils import (
    mime_type,
    exporter,
    read_json,
    write_json,
    validate_url,
)
from eight_mile.progress import create_progress_bar

logger = logging.getLogger('mead.layers')

__all__ = []
export = exporter(__all__)


def _verify_file(file_loc):
    # dropbox doesn't give 404 in case the file does not exist, produces an HTML. The actual files are never HTMLs.
    if not os.path.exists(file_loc):
        return False

    if os.path.isfile(file_loc) and mime_type(file_loc) == "text/html":
        return False

    return True


@export
def delete_old_copy(file_name):
    if os.path.exists(file_name):
        if os.path.isfile(file_name):
            os.remove(file_name)
        else:
            shutil.rmtree(file_name)
    return file_name


@export
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



@export
def extract_tar(file_loc):
    import tarfile
    temp_file = delete_old_copy("{}.1".format(file_loc))
    with tarfile.open(file_loc, "r") as tar_ref:
        tar_ref.extractall(temp_file)
    if len(os.listdir(temp_file)) != 1:
        raise RuntimeError("tar extraction unsuccessful")
    return os.path.join(temp_file, os.listdir(temp_file)[0])


@export
def extract_zip(file_loc):
    temp_file = delete_old_copy("{}.1".format(file_loc))
    with zipfile.ZipFile(file_loc, "r") as zip_ref:
        zip_ref.extractall(temp_file)
    return temp_file


@export
def extractor(filepath, cache_dir, extractor_func):
    with open(filepath, 'rb') as f:
        sha1 = hashlib.sha1(f.read()).hexdigest()
    logger.info("extracting file..")
    path_to_save = filepath if extractor_func is None else extractor_func(filepath)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    path_to_save_sha1 = os.path.join(cache_dir, sha1)
    delete_old_copy(path_to_save_sha1)
    shutil.move(path_to_save, path_to_save_sha1)
    logger.info("downloaded data saved in {}".format(path_to_save_sha1))
    return path_to_save_sha1


@export
def get_file_or_url(file, cache=None):
    """
    Return itself if its a file, else downloaded.  If cache, use that
    :param file: The file name
    :param cache: A cache location, If `None`, just download
    :return: The proper filename
    """

    if validate_url(file):
        logger.info(f'Downloading {file}')
        if cache:
            file = SingleFileDownloader(file, cache).download()
        else:
            file, _ = urlretrieve(file)
    return file


@export
def open_file_or_url(file, mode='r', cache=None):
    """
    Return handle to a file, if a url, download first. If cache, use that
    :param file: The file name
    :param model: The open mode
    :param cache: A cache location, If `None`, just download
    :return: The handle
    """
    return open(get_file_or_url(file, cache), mode)


@export
def web_downloader(url, path_to_save=None):
    # Use a class to simulate the nonlocal keyword in 2.7
    class Context: pg = None

    def _report_hook(count, block_size, total_size):
        if Context.pg is None:
            length = int((total_size + block_size - 1) / float(block_size)) if total_size != -1 else 1
            Context.pg = create_progress_bar(length)
        Context.pg.update()

    if not path_to_save:
        path_to_save = "/tmp/data.dload-{}".format(os.getpid())
    try:
        path_to_save, _ = urlretrieve(url, path_to_save, reporthook=_report_hook)
        Context.pg.done()
    except Exception as e:  # this is too broad but there are too many exceptions to handle separately
        raise RuntimeError("failed to download data from [url]: {} [to]: {}".format(url, path_to_save))
    return path_to_save



@export
def is_file_correct(file_loc, data_dcache=None, key=None):
    """check if the file location mentioned in the json file is correct, i.e.,
    exists and not corrupted. This is needed when the direct download link/ path for a file
    changes and the user is unaware. This is not tracked by sha1 either. If it returns False, delete the corrupted file.
    Additionally, if the file location is a URL, i.e. exists in the cache, delete it so that it can be re-downloaded.

    Keyword arguments:
    file_loc -- location of the file
    data_dcache -- data download cache location (default None, for local system file paths)
    key -- URL for download (default None, for local system file paths)
    """
    if _verify_file(file_loc):
        return True
    # Some files are prefixes (the datasset.json has `train` and the data has `train.fr` and `train.en`)
    dir_name = os.path.dirname(file_loc)
    # When we are using this for checking embeddings file_loc is a url so we need this check.
    if os.path.exists(dir_name):
        files = [os.path.join(dir_name, f) for f in os.listdir(dir_name) if os.path.join(dir_name, f).startswith(file_loc)]
        if files and all(_verify_file(f) for f in files):
            return True
    delete_old_copy(file_loc)
    if key is not None:  # cache file validation
        update_cache(key, data_dcache)
    return False


@export
def is_dir_correct(dir_loc, dataset_desc, data_dcache, key, ignore_file_check=False):
    """check if the directory extracted from the zip location mentioned in the datasets json file is correct, i.e.,
    all files inside exist and are not corrupted. If not, we will update the cache try to re-download them.

    Keyword arguments:
    dir_loc -- location of the directory
    dataset_desc -- to know the individual file locations inside the directory
    data_dcache -- data download cache location
    key -- URL for download
    ignore_file_check --to handle enc_dec datasets, see later.
    """

    if not os.path.exists(dir_loc) or not os.path.isdir(dir_loc):
        update_cache(key, data_dcache)
        return False
    if ignore_file_check:  # for enc_dec tasks there's no direct downloads
        return True
    files = [os.path.join(dir_loc, dataset_desc[k]) for k in dataset_desc if k.endswith("_file")]
    for f in files:
        if not is_file_correct(f, key, data_dcache):
            return False
    return True


DATA_CACHE_CONF = "data-cache.json"


@export
def update_cache(key, data_download_cache):
    dcache = read_json(os.path.join(data_download_cache, DATA_CACHE_CONF))
    if key not in dcache:
        return
    del dcache[key]
    write_json(dcache, os.path.join(data_download_cache, DATA_CACHE_CONF))




@export
class Downloader:
    ZIPD = {'application/gzip': extract_gzip, 'application/zip': extract_zip}

    def __init__(self, data_download_cache, cache_ignore):
        super().__init__()
        self.cache_ignore = cache_ignore
        self.data_download_cache = data_download_cache

    def download(self):
        pass


@export
class SingleFileDownloader(Downloader):
    def __init__(self, dataset_file, data_download_cache, cache_ignore=False):
        super().__init__(data_download_cache, cache_ignore)
        self.dataset_file = dataset_file
        self.data_download_cache = data_download_cache

    def download(self):
        file_loc = self.dataset_file
        if is_file_correct(file_loc):
            return file_loc
        elif validate_url(file_loc):  # is it a web URL? check if exists in cache
            url = file_loc
            dcache_path = os.path.join(self.data_download_cache, DATA_CACHE_CONF)
            dcache = read_json(dcache_path)
            if url in dcache and is_file_correct(dcache[url], self.data_download_cache, url) and not self.cache_ignore:
                logger.info("file for {} found in cache, not downloading".format(url))
                return dcache[url]
            else:  # download the file in the cache, update the json
                cache_dir = self.data_download_cache
                logger.info("using {} as data/embeddings cache".format(cache_dir))
                temp_file = web_downloader(url)
                dload_file = extractor(filepath=temp_file, cache_dir=cache_dir,
                                       extractor_func=Downloader.ZIPD.get(mime_type(temp_file), None))
                dcache.update({url: dload_file})
                write_json(dcache, os.path.join(self.data_download_cache, DATA_CACHE_CONF))
                return dload_file
        raise RuntimeError("the file [{}] is not in cache and can not be downloaded".format(file_loc))


@export
class AddonDownloader(Downloader):
    ADDON_SUBPATH = 'addons'
    """Grab addons and write them to the download cache
    """
    def __init__(self, dataset_file, data_download_cache, cache_ignore=False):
        super().__init__(data_download_cache, cache_ignore)
        self.dataset_file = dataset_file
        self.data_download_cache = data_download_cache

    def download(self):
        file_loc = self.dataset_file
        if is_file_correct(file_loc):
            return file_loc
        elif validate_url(file_loc):  # is it a web URL? check if exists in cache
            url = file_loc
            dcache_path = os.path.join(self.data_download_cache, DATA_CACHE_CONF)
            dcache = read_json(dcache_path)
            # If the file already exists in the cache
            if url in dcache and is_file_correct(dcache[url], self.data_download_cache, url) and not self.cache_ignore:
                logger.info("file for {} found in cache, not downloading".format(url))
                return dcache[url]
            # Otherwise, we want it to be placed in ~/.bl-cache/addons
            else:  # download the file in the cache, update the json
                cache_dir = self.data_download_cache
                addon_path = os.path.join(cache_dir, AddonDownloader.ADDON_SUBPATH)
                if not os.path.exists(addon_path):
                    os.makedirs(addon_path)
                path_to_save = os.path.join(addon_path, os.path.basename(file_loc))
                logger.info("using {} as data/addons cache".format(cache_dir))
                web_downloader(url, path_to_save)
                dcache.update({url: path_to_save})
                write_json(dcache, os.path.join(self.data_download_cache, DATA_CACHE_CONF))
                return path_to_save
        raise RuntimeError("the file [{}] is not in cache and can not be downloaded".format(file_loc))

def _update_md(dataset_description_map, download_dir):
    updated = {k: v for k, v in dataset_description_map.items()}
    if download_dir:
        updated.update({k: os.path.join(download_dir, dataset_description_map[k]) for k in dataset_description_map.keys() if k.endswith('_file')})
    return updated

@export
class DataDownloader(Downloader):
    def __init__(self, dataset_desc, data_download_cache, enc_dec=False, cache_ignore=False):
        super().__init__(data_download_cache, cache_ignore)
        self.dataset_desc = dataset_desc
        self.data_download_cache = data_download_cache
        self.enc_dec = enc_dec

    def download(self):
        dload_bundle = self.dataset_desc.get("download", None)
        if dload_bundle is not None:  # download a zip/tar/tar.gz directory, look for train, dev test files inside that.
            dcache_path = os.path.join(self.data_download_cache, DATA_CACHE_CONF)
            dcache = read_json(dcache_path)
            if dload_bundle in dcache and \
                    is_dir_correct(dcache[dload_bundle], self.dataset_desc, self.data_download_cache, dload_bundle,
                                   self.enc_dec) and not self.cache_ignore:
                download_dir = dcache[dload_bundle]
                logger.info("files for {} found in cache, not downloading".format(dload_bundle))
                updated = _update_md(self.dataset_desc, download_dir)
                return updated
            else:  # try to download the bundle and unzip
                if not validate_url(dload_bundle):
                    raise RuntimeError("can not download from the given url")
                else:
                    cache_dir = self.data_download_cache
                    temp_file = web_downloader(dload_bundle)

                    download_dir = extractor(filepath=temp_file, cache_dir=cache_dir,
                                             extractor_func=Downloader.ZIPD.get(mime_type(temp_file), None))
                    if "sha1" in self.dataset_desc:
                        if os.path.split(download_dir)[-1] != self.dataset_desc["sha1"]:
                            raise RuntimeError("The sha1 of the downloaded file does not match with the provided one")
                    dcache.update({dload_bundle: download_dir})
                    write_json(dcache, os.path.join(self.data_download_cache, DATA_CACHE_CONF))
                    updated = _update_md(self.dataset_desc, download_dir)
                    return updated
        else:  # we have download links to every file or they exist
            updated = _update_md(self.dataset_desc, None)
            if not self.enc_dec:
                updated.update({k: SingleFileDownloader(self.dataset_desc[k], self.data_download_cache).download()
                        for k in self.dataset_desc if k.endswith("_file") and self.dataset_desc[k]})
            return updated


@export
class EmbeddingDownloader(Downloader):
    def __init__(self, embedding_file, embedding_dsz, embedding_sha1, data_download_cache, cache_ignore=False, unzip_file=True):
        super().__init__(data_download_cache, cache_ignore)
        self.embedding_file = embedding_file
        self.embedding_key = embedding_dsz
        self.data_download_cache = data_download_cache
        self.unzip_file = unzip_file
        self.sha1 = embedding_sha1

    @staticmethod
    def _get_embedding_file(loc, key):
        if os.path.isfile(loc):
            logger.info("embedding file location: {}".format(loc))
            return loc
        else:  # This is a directory, return the actual file
            files = [x for x in os.listdir(loc) if str(key) in x]
            if len(files) == 0:
                raise RuntimeError("No embedding file found for the given key [{}]".format(key))
            elif len(files) > 1:
                logger.info("multiple embedding files found for the given key [{}], choosing {}".format(key, files[0]))
            embed_file_loc = os.path.join(loc, files[0])
            return embed_file_loc

    def download(self):
        if is_file_correct(self.embedding_file):
            logger.info("embedding file location: {}".format(self.embedding_file))
            return self.embedding_file
        dcache_path = os.path.join(self.data_download_cache, DATA_CACHE_CONF)
        dcache = read_json(dcache_path)
        if self.embedding_file in dcache and not self.cache_ignore:
            download_loc = dcache[self.embedding_file]
            logger.info("files for {} found in cache".format(self.embedding_file))
            return self._get_embedding_file(download_loc, self.embedding_key)
        else:  # try to download the bundle and unzip
            url = self.embedding_file
            if not validate_url(url):
                raise RuntimeError("can not download from the given url")
            else:
                cache_dir = self.data_download_cache
                temp_file = web_downloader(url)
                unzip_fn = Downloader.ZIPD.get(mime_type(temp_file)) if self.unzip_file else None
                download_loc = extractor(filepath=temp_file, cache_dir=cache_dir,
                                         extractor_func=unzip_fn)
                if self.sha1 is not None:
                    if os.path.split(download_loc)[-1] != self.sha1:
                        raise RuntimeError("The sha1 of the downloaded file does not match with the provided one")
                dcache.update({url: download_loc})
                write_json(dcache, os.path.join(self.data_download_cache, DATA_CACHE_CONF))
                return self._get_embedding_file(download_loc, self.embedding_key)


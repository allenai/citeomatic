import errno
import io
import json
import logging
import os
import pickle
import tarfile
import typing
import tempfile
import hashlib
import subprocess
from os.path import abspath, dirname, join
from gzip import GzipFile

import arrow

ROOT = abspath(dirname(dirname(dirname(__file__))))


class S3FileNotFoundError(FileNotFoundError):
    pass


def _expand(filename):
    return os.path.expanduser(filename)


def _is_okay_cache_dir(name):
    if os.path.exists(name) or os.system('mkdir -p %s' % name) == 0:
        return name


def _cache_dir():
    # Try using a shared data drive if it's available
    dirs = [
        '/data/cache/s2-research',
        '/tmp/s2-research-cache/',
        '/tmp/',
    ]
    for name in dirs:
        if _is_okay_cache_dir(name):
            logging.info('Using %s for caching', name)
            return name

    assert False, 'Failed to find suitable cache directory'


def last_modified(filename):
    if filename.startswith('s3://'):
        return S3File.last_modified(filename)
    else:
        if os.path.exists(filename):
            return arrow.get(os.path.getmtime(filename))
        else:
            return None


class StreamingS3File(object):
    def __init__(self, name, mode, encoding):
        assert 'w' not in mode and 'a' not in mode, 'Streaming writes not supported.'
        key = _s3_key(name)
        if key is None:
            raise FileNotFoundError(name)

        streaming_file = key.get()['Body']

        def _readinto(buf):
            bytes_read = streaming_file.read(len(buf))
            buf[:len(bytes_read)] = bytes_read
            return len(bytes_read)

        streaming_file.readinto = _readinto
        streaming_file.readable = lambda: True
        streaming_file.writable = lambda: False
        streaming_file.seekable = lambda: False
        streaming_file.closeable = lambda: False
        streaming_file.closed = False
        streaming_file.flush = lambda: 0

        self._file = io.BufferedReader(streaming_file, buffer_size=512000)
        if encoding is not None or 't' in mode:
            # The S3 file interface from boto doesn't conform to the standard python file interface.
            # Add dummy methods to make the text wrapper happy.
            self._file = io.TextIOWrapper(self._file, encoding=encoding)

    def readable(self):
        return True

    def writeable(self):
        return False

    def seekable(self):
        return False

    def closeable(self):
        return False

    @property
    def closed(self):
        return False

    def flush(self):
        return 0

    def read(self, *args):
        return self._file.read(*args)

    def readline(self):
        return self._file.readline()

    def close(self):
        return self._file.close()

    def seekable(self):
        return False

    def __enter__(self):
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()


def cache_file(name):
    if not name.startswith('s3://'):
        return name

    s3_last_modified = last_modified(name)
    cleaned_name = name[5:].replace('/', '_')
    target_filename = os.path.join(_cache_dir(), cleaned_name)
    if os.path.exists(target_filename):
        if s3_last_modified is None or last_modified(
            target_filename
        ) >= s3_last_modified:
            return target_filename

    logging.info('Cache file for %s does not exist, copying.', name)
    parse = _parse_s3_location(name)
    retcode = subprocess.call(
        'aws s3api get-object --bucket "%s" --key "%s" "%s.tmp.%d" --request-payer=requester'
        % (parse['bucket'], parse['key'], target_filename, os.getpid()),
        stdout=subprocess.DEVNULL,
        shell=True
    )
    if retcode != 0:
        raise FileNotFoundError('Failed to copy %s' % name)
    assert os.system(
        'mv "%s.tmp.%d" "%s"' %
        (target_filename, os.getpid(), target_filename)
    ) == 0
    assert os.system('chmod 777 "%s"' % (target_filename)) == 0
    return target_filename


def s3_location_to_object(path):
    s3 = boto3.resource('s3')

    parse = _parse_s3_location(path)
    bucket_name = parse['bucket']
    key = parse['key']
    return s3.Object(bucket_name, key)


def _parse_s3_location(path):
    logging.debug('Parsing path %s' % path)
    if not path.startswith('s3://'):
        raise ValueError('s3 location must start with s3://')

    path = path[5:]
    parts = path.split('/', 1)
    if len(parts) == 1:
        bucket = parts[0]
        key = None
    else:
        bucket, key = parts

    return {'bucket': bucket, 'key': key}


# Yield S3 objects with a given prefix.
def iterate_s3_objects(path, max_files=None):
    import boto3

    # Check if path exists on S3
    if path.startswith('s3://'):
        parsed_location = _parse_s3_location(path)
        bucket = parsed_location['bucket']
        folder_key = parsed_location['key']

        s3 = boto3.resource('s3')
        client = boto3.client('s3')
        s3_bucket = s3.Bucket(bucket)

        if max_files:
            s3_obj_iterator = \
                s3_bucket.objects.filter(Prefix=folder_key, RequestPayer='requester').limit(max_files)
        else:
            s3_obj_iterator = s3_bucket.objects.filter(
                Prefix=folder_key, RequestPayer='requester'
            ).all()

        yield from s3_obj_iterator


# Yield s3 filenames with a given prefix.
def iterate_s3_files(path_prefix, max_files=None):
    # return the full name of each file.
    for s3_object in iterate_s3_objects(path_prefix, max_files):
        yield 's3://{}/{}'.format(s3_object.bucket_name, s3_object.key)


# Deprecated. For backward compatibility.
def iterate_s3(path):
    yield from iterate_s3_objects(path)


def iterate_s3_files(path_prefix, max_files=None):
    """Yield s3 filenames with a given prefix."""
    # return the full name of each file.
    for s3_object in iterate_s3_objects(path_prefix, max_files):
        yield 's3://{}/{}'.format(s3_object.bucket_name, s3_object.key)


def iterate_files(path_prefix: str) -> typing.Iterable[str]:
    """Yield filenames with a given prefix."""
    if path_prefix.startswith('s3://'):
        yield from iterate_s3_files(path_prefix)
    else:
        for (root, directories, filenames) in os.walk(path_prefix):
            for filename in filenames:
                yield os.path.join(root, filename)


class S3File(object):
    def __init__(self, name, mode, encoding):
        self.name = name
        self.mode = mode
        self.encoding = encoding

        if 'r' in mode:
            self._local_name = self._cache()
            self._local_file = io.open(self._local_name, mode)
        else:
            prefix = self.name.split('//')[1].replace('/', '_')
            self._local_name = join(_cache_dir(), '.tmp_' + prefix)
            self._local_file = io.open(
                self._local_name, mode=mode, encoding=encoding
            )

    @staticmethod
    def last_modified(filename):
        key = _s3_key(filename)
        if key is None:
            return None
        return arrow.get(key.last_modified)

    def flush(self):
        logging.info('Syncing "%s" to S3' % self.name)
        self._local_file.flush()
        assert os.system(
            'aws s3 cp "%s" "%s"' % (self._local_name, self.name)
        ) == 0

    def write(self, *args):
        return self._local_file.write(*args)

    def read(self, *args):
        return self._local_file.read(*args)

    def read_lines(self, *args):
        return self._local_file.read(*args)

    def _cache(self):
        return cache_file(self.name)

    def seekable(self):
        return True

    def close(self):
        if 'w' in self.mode or 'a' in self.mode:
            self.flush()
            os.unlink(self._local_name)
        else:
            self._local_file.close()

    def __enter__(self):
        return self._local_file

    def __exit__(self, type, value, traceback):
        self.close()


def _gzip_file(fileobj, mode, encoding):
    def _fix_fileobj(gzip_file):
        """
        Terrible hack to ensure that GzipFile actually calls close on the fileobj passed into it.
        """
        gzip_file.myfileobj = gzip_file.fileobj
        return gzip_file

    if 't' in mode or encoding is not None:
        mode = mode.replace('t', '')
        f = _fix_fileobj(GzipFile(fileobj=fileobj, mode=mode))
        return io.TextIOWrapper(f, encoding=encoding)
    else:
        f = _fix_fileobj(GzipFile(fileobj=fileobj, mode=mode))
        if 'r' in mode:
            return io.BufferedReader(f)
        else:
            return io.BufferedWriter(f)


def _bzip_file(fileobj, mode, encoding):
    import bz2
    if 't' in mode:
        bz2_file = bz2.BZ2File(fileobj, mode=mode.replace('t', 'b'))
        bz2_file._closefp = True
        return io.TextIOWrapper(bz2_file, encoding)
    else:
        bz2_file = bz2.BZ2File(fileobj, mode=mode)
        bz2_file._closefp = True
        return bz2_file


def data_file(name):
    """Read a data file from the source repository."""
    return os.path.join(ROOT, 'data', name)


def test_file(caller_filename: str, name: str) -> str:
    curdir = abspath(caller_filename)
    while curdir != '/':
        fname = os.path.join(curdir, 'testdata', name)
        if os.path.exists(fname):
            return fname

        curdir = os.path.dirname(curdir)

    raise FileNotFoundError('Failed to find testdata file: %s' % name)


def slurp(filename, mode='r', encoding=None):
    """Read all content from `filename`"""
    with open(
        _expand(filename), mode=mode, encoding=encoding, streaming=True
    ) as f:
        return f.read()


def read_json(filename):
    """Read  JSON from `filename`."""
    with open(_expand(filename), 'rt') as f:
        return json.load(f)


def write_json(filename, obj, indent=None, sort_keys=None):
    """Write JSON to `filename`"""
    with open(_expand(filename), 'w') as f:
        json.dump(obj, f, indent=indent, sort_keys=sort_keys)


def write_json_atomic(filename, obj, indent=None, sort_keys=None):
    """Write JSON to `filename` such that `filename` never exists in a partially written state."""
    filename = _expand(filename)
    if filename.startswith('s3://'):
        write_json(
            filename, obj, indent, sort_keys
        )  # s3 operations are already atomic
    with tempfile.NamedTemporaryFile(
        'w', dir=os.path.dirname(filename), delete=False
    ) as f:
        json.dump(obj, f, indent=indent, sort_keys=sort_keys)
        tempname = f.name
    os.rename(tempname, filename)


def read_pickle(filename, streaming=False):
    """Read  pickled data from `name`."""
    with open(_expand(filename), 'rb', streaming=streaming) as f:
        return pickle.load(f)


def write_pickle(filename, obj):
    with open(_expand(filename), 'wb') as f:
        pickle.dump(obj, f, -1)


def write_file(filename, value: typing.Union[bytes, str], mode='w'):
    with open(_expand(filename), mode) as f:
        f.write(value)


def write_file_if_not_exists(
    filename, value: typing.Union[bytes, str], mode='w'
):
    if os.path.exists(_expand(filename)):
        return

    write_file(filename, value, mode)


def write_file_atomic(
    filename: str, value: typing.Union[bytes, str], mode='w'
) -> None:
    if filename.startswith('s3://'):
        write_file(filename, value, mode)
    else:
        with tempfile.NamedTemporaryFile(
            'w', dir=os.path.dirname(filename), delete=False
        ) as f:
            f.write(value)
            tempname = f.name
        os.rename(tempname, filename)


def read_lines(filename, comment=None, streaming=False):
    """
    Read all non-blank lines from `filename`.

    Skip any lines that begin the comment character.
    :param filename: Filename to read from.
    :param comment: If defined, ignore lines starting with this text.
    :return:
    """
    with open(_expand(filename), 'rt', streaming=streaming) as f:
        for l in f:
            if comment and not l.startswith(comment):
                continue
            yield l.strip()


def read_json_lines(filename, streaming=False):
    for line in read_lines(filename, streaming=streaming):
        yield json.loads(line)


def exists(filename):
    return last_modified(filename) is not None


def open(filename, mode='rb', encoding=None, **kw):
    """
    Open `filename` for reading.  If filename is compressed with a known format,
    it will be transparently decompressed.

    Optional keyword args:

    `streaming`: if true, remote files will be streamed directly; no local cache
    will be generated.

    `no_zip`: do not try to automatically decompress the input file
    """
    if filename.endswith('.gz') and 'no_decompress' not in kw:
        if 'r' in mode:
            target_mode = 'rb'
        else:
            target_mode = 'wb'

        target = open(
            filename,
            no_decompress=True,
            mode=target_mode,
            encoding=None,
            **kw
        )
        return _gzip_file(target, mode, encoding)

    if filename.endswith('.bz2') and 'no_decompress' not in kw:
        if 'r' in mode:
            target_mode = 'rb'
        else:
            target_mode = 'wb'

        target = open(
            filename,
            no_decompress=True,
            mode=target_mode,
            encoding=None,
            **kw
        )
        return _bzip_file(target, mode, encoding)

    if filename.startswith('s3://'):
        if kw.get('streaming', False):
            return StreamingS3File(filename, mode, encoding)
        else:
            return S3File(filename, mode, encoding)

    import io
    return io.open(filename, mode, encoding=encoding)


def safe_makedirs(dir_path: str) -> None:
    """Create a directory if it doesn't already exist, avoiding race conditions if called from multiple processes."""
    dir_path = _expand(dir_path)
    try:
        os.makedirs(dir_path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(dir_path):
            pass
        else:
            raise


def copy(src: str, dst: str) -> None:
    """Copy src to dst."""
    src = _expand(src)
    dst = _expand(dst)
    with open(src, 'rb') as src_f, open(dst, 'wb') as dst_f:
        while True:
            chunk = src_f.read(4096)
            if chunk is None or len(chunk) == 0:
                break
            dst_f.write(chunk)


def extract_tarfile_from_bytes(b: bytes, dst: str, mode='r') -> None:
    seekable_f = io.BytesIO(b)
    safe_makedirs(os.path.dirname(dst))
    with tarfile.open(fileobj=seekable_f, mode=mode) as t:
        t.extractall(path=dst)


def extract_tarfile(src: str, dst: str, streaming=True) -> None:
    """Extract a tarfile at 'src' to 'dst'."""
    src = _expand(src)
    dst = _expand(dst)
    with open(src, mode='rb', streaming=streaming) as f:
        b = f.read()
    extract_tarfile_from_bytes(b, dst)


def compute_sha1(filename: str, buf_size=int(1e6)) -> str:
    sha1 = hashlib.sha1()
    with open(filename, 'rb') as f:
        while True:
            data = f.read(buf_size)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


class SetJsonEncoder(json.JSONEncoder):
    """Simple JSONEncoder that encodes sets as lists."""

    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)
        return json.JSONEncoder.default(self, obj)


class JsonFile(object):
    '''
    A flat text file where each line is one json object

    # to read though a file line by line
    with JsonFile('file.json', 'r') as fin:
        for line in fin:
            # line is the deserialized json object
            pass


    # to write a file object by object
    with JsonFile('file.json', 'w') as fout:
        fout.write({'key1': 5, 'key2': 'token'})
        fout.write({'key1': 0, 'key2': 'the'})
    '''

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __iter__(self):
        for line in self._file:
            yield json.loads(line)

    def write(self, item):
        item_as_json = json.dumps(item, ensure_ascii=False)
        encoded = '{0}\n'.format(item_as_json)
        self._file.write(encoded)

    def __enter__(self):
        self._file = open(*self._args, **self._kwargs)
        self._file.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.__exit__(exc_type, exc_val, exc_tb)


class GzipJsonFile(JsonFile):
    '''
    A gzip compressed JsonFile.  Usage is the same as JsonFile
    '''

    def __enter__(self):
        self._file = GzipFile(*self._args, **self._kwargs)
        self._file.__enter__()
        return self

    def __iter__(self):
        for line in self._file:
            yield json.loads(line.decode('utf-8', 'ignore'))

    def write(self, item):
        item_as_json = json.dumps(item, ensure_ascii=False)
        encoded = '{0}\n'.format(item_as_json).encode('utf-8', 'ignore')
        self._file.write(encoded)

# ==================================================================================
# Copyright (c) 2016, Brain Corporation
#
# This software is released under Creative Commons
# Attribution-NonCommercial-ShareAlike 3.0 (BY-NC-SA) license.
# Full text available here in LICENSE.TXT file as well as:
# https://creativecommons.org/licenses/by-nc-sa/3.0/us/legalcode
#
# In summary - you are free to:
#
#    Share - copy and redistribute the material in any medium or format
#    Adapt - remix, transform, and build upon the material
#
# The licensor cannot revoke these freedoms as long as you follow the license terms.
#
# Under the following terms:
#    * Attribution - You must give appropriate credit, provide a link to the
#                    license, and indicate if changes were made. You may do so
#                    in any reasonable manner, but not in any way that suggests
#                    the licensor endorses you or your use.
#    * NonCommercial - You may not use the material for commercial purposes.
#    * ShareAlike - If you remix, transform, or build upon the material, you
#                   must distribute your contributions under the same license
#                   as the original.
#    * No additional restrictions - You may not apply legal terms or technological
#                                   measures that legally restrict others from
#                                   doing anything the license permits.
# ==================================================================================
import ssl
import socket
import sys
import traceback
import os
import logging
import re
import random
import hashlib
import errno
import shutil


def get_S3_credentials():
    """
    The function will attempt to load credentials from ~/.aws/credentials
    (as well as environment variables, boto config files and other places)

    If succesfull will return a credentials object.

    :return:
    """
    from boto.provider import Provider
    from boto.sts.credentials import Credentials
    provider = Provider('aws')
    if provider.access_key and provider.secret_key:
        cred = Credentials()
        cred.access_key = provider.access_key
        cred.secret_key = provider.secret_key
        return cred
    else:
        logging.getLogger(__name__).info("Could not load permanent credentials")
        return None


class Storage(object):

    _pattern = re.compile("[a-zA-Z0-9._\\-/]*$")

    def __init__(self, local_storage_folder="~/PVM_data", S3_bucket="technical.services.braincorporation.net"):
        """
        Initialize storage object. All the elements will by stored in the local_storage_folder.

        If in addition an S3 bucket and credentials are given, the local storage will act as local cache
        for S3.
        :param local_storage_folder: A folder where the files are stored/mirrored
        :param S3_bucket: Optional S3 bucket where the data will be stored
        :param S3_credentials: Credentials to access the bucket
        :return:
        """
        if S3_bucket is not None:
            S3_credentials = get_S3_credentials()
            if S3_credentials is not None and S3_credentials.access_key and S3_credentials.secret_key:
                import boto
                logging.getLogger('boto').setLevel(logging.CRITICAL)
                self.connector = boto.connect_s3(S3_credentials.access_key,
                                                 S3_credentials.secret_key,
                                                 security_token=S3_credentials.session_token,
                                                 calling_format='boto.s3.connection.OrdinaryCallingFormat')
                # getting bucket with disabled validation to avoid connection error:'S3ResponseError:403 URLBlocked'
                self.bucket = self.connector.get_bucket(S3_bucket, validate=False)
            else:
                self.bucket = None
        else:
            self.bucket = None
        self.local_storage = os.path.expanduser(local_storage_folder)
        if not os.path.exists(self.local_storage):
            os.mkdir(self.local_storage)

    def is_valid_key(self, key):
        return self._pattern.match(key) is not None

    def exists(self, key):
        """
        Do not check existence of a folder because there are no folders in S3
        """
        assert self.is_valid_key(key)
        from boto.exception import S3ResponseError
        try:
            k = self.bucket.get_key(key)
            return k is not None
        except S3ResponseError:
            # When using a guest account, checking whether an object exists
            # will result in a 503 Forbidden exception if it does not exist
            return False

    def _ensure_containing_folder_exists(self, filepath):
        """
        Make sure that the folder containing filepath exists already to avoid
        errors when creating the file.
        """
        directory = os.path.dirname(filepath)
        try:
            os.makedirs(directory)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise

    def get(self, path, to_path=None, force=False):
        """
        Get a file from local/remote storage. The file will get downloaded from S3 if bucket is in use, to a local storage
        directory. Path to the file in local storage directory will be returned.
        :param path:
        :param to_path:
        :param force:
        :return:
        """
        logging.info("Storage received a request to get file %s" % path)
        if self.bucket is None:
            file_p = os.path.join(self.local_storage, path)
            logging.info("Loading file %s" % file_p)
            if not os.path.exists(file_p):
                # maybe then such file actually exist in the file system?
                if not os.path.exists(path):
                    logging.error("File %s does not exist!" % path)
                    raise Exception("Requested file %s does not exist" % path)
                else:
                    return path
            else:
                return file_p
        else:
            if to_path is None:
                dest_path = os.path.join(self.local_storage, path)
            else:
                dest_path = to_path
            if os.path.isfile(dest_path):
                if self.exists(path) and force:
                    os.remove(dest_path)
                elif force:
                    raise Exception("Forcing download but the original file does not exist on S3")
                else:
                    return dest_path
            try:
                if self.exists(path) and not path.endswith("/"):
                    while True:
                        try:
                            self.download(to_file=dest_path, key=path, progress_callback=self._progress_download)
                            break
                        except ssl.SSLError, socket.error:
                            logging.error("Download timeout, trying again! file %s" % path)
                            continue
                    logging.info("Downloaded file %s to %s" % (path, dest_path))
                    return dest_path
                else:
                    if not os.path.exists(path):
                        logging.error("File %s does not exist!" % path)
                        raise Exception("Requested file %s does not exist" % path)
                    else:
                        logging.info("Using file found in the local file system. %s" % path)
                        return path
            except:
                traceback.print_exc(file=sys.stdout)
                return None

    def put(self, path=None, from_path=None, to_folder=None, overwrite=False):
        """
        Put a file to local/remot storage.
        If using an S3 bucket the file will be uploaded. If not it will be copied to local storage

        :param path: path (key) within the local cache structure (same are remote S3 path)
        :type path: str
        :param from_path: (optional) path to the file that should be uploaded
        :type from path: str
        :param to_folder: (optional) remote directory to which the file needs to be uploaded
        :type to_folder: str
        :param overwrite: if True a file will be overwritten if it already exists remotely
        :type overwrite: bool
        :return: True if successful, None if error
        """
        if path == "/":
            path = ""
        # Just copy to the right directory
        if path is None:
            dst = os.path.join(self.local_storage, to_folder)
            dst = os.path.join(dst, os.path.basename(from_path))
        else:
            dst = os.path.join(self.local_storage, path)
        if os.path.isfile(dst) and not overwrite:
            logging.error("File %s already exists, not overwriting" % dst)
            return None
        self._ensure_containing_folder_exists(dst)
        if not (os.path.isfile(dst) and os.path.samefile(from_path, dst)):
            shutil.copy2(from_path, dst)
        if self.bucket is not None:
            if from_path is None:
                source_path = os.path.join(self.local_storage, path)
            else:
                source_path = from_path
            if to_folder is None:
                to_folder = os.path.dirname(path)
            if not os.path.isfile(source_path):
                logging.error("File %s does not exists" % source_path)
                return None
            while True:
                try:
                    self.upload(from_file=source_path,
                                to_folder=to_folder,
                                progress_callback=self._progress_upload,
                                overwrite=overwrite)
                    break
                except ssl.SSLError, socket.error:
                    logging.error("Upload timeout, trying again!, file %s" % source_path)
                    continue
            logging.info("Uploaded file %s to folder %s" % (source_path, to_folder))
            return True

    def download(self, to_file, key, progress_callback=None):
        """
        Download a file from S3 to a local storage

        :param to_file: filename in which to store the downloaded file (str)
        :param key: the key to download from the bucket (str)
        :param progress_callback: callback function printing progress
        :return:
        """
        from boto.s3.key import Key
        k = Key(self.bucket, key)
        self._ensure_containing_folder_exists(to_file)
        tmpfile = to_file + '.tmp.%s' % hashlib.sha1(str(random.getrandbits(256))).hexdigest()
        try:
            k.get_contents_to_filename(filename=tmpfile, cb=progress_callback, num_cb=100)
        except:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)
            raise
        # download is successful, move the temporary file to desired location
        os.rename(tmpfile, to_file)

    def upload(self, from_file, to_folder, overwrite=False, progress_callback=None):
        """
        Upload a file to S3 bucket.
        S3 doen't have concept of folder so /path/name.txt is just a long filename.
        It means that 'folders' will be created automatically if they do not exists

        :param from_file: path to the file being uploaded
        :param to_folder: S3 "folder" to which the file should go.
        :param overwrite: overwrite if True
        :param progress_callback: callback function printing progress
        :return:
        """
        from boto.s3.key import Key
        custom_key_name = os.path.basename(from_file)
        keyname = os.path.join(to_folder, custom_key_name)
        assert self.is_valid_key(keyname)
        # check that file exists
        if not overwrite and self.exists(keyname):
            raise Exception('Can not upload, file %s already exists on amazon' %
                            keyname)

        logging.getLogger(__name__).info('UI:Uploading to "%s"...' % keyname)
        k = Key(self.bucket, keyname)
        k.set_contents_from_filename(from_file, cb=progress_callback, num_cb=100)
        return True

    def _progress_download(self, elapsed, total):
        """
        Prints the download progress
        :param elapsed:
        :param total:
        :return:
        """
        print "Downloading %d %% \r" % (elapsed*100/max(total, 1)),
        sys.stdout.flush()

    def _progress_upload(self, elapsed, total):
        """
        Prints the upload progress
        :param elapsed:
        :param total:
        :return:
        """
        print "Uploading %d %% \r" % (elapsed*100/max(total, 1)),
        sys.stdout.flush()

    def list_path(self, path):
        """
        List all the keys in the remote path

        :param path: path (key) in the local or remote storage
        :type path: str
        :return: list of keys
        :rtype: list
        """
        if path == "/":
            path = ""
        result = []
        if self.bucket is None:
            for root, dirs, files in os.walk(self.local_storage, topdown=False):
                for name in files:
                    result.append(os.path.join(root, name))
        else:
            assert self.is_valid_key(path)
            keys_list = self.bucket.list(prefix=path)
            result = [key.name for key in keys_list if os.path.normpath(key.name) != os.path.normpath(path)]
        return result

    def remove(self, key, check_existence=True):
        """
        Remove a file from local/remote storage

        :param key:
        :param check_existence:
        """
        if self.bucket is None:
            if check_existence and not os.path.exists(os.path.join(self.local_storage, key)):
                raise Exception('Can not remove non-existent key %s' % key)
            os.remove(os.path.join(self.local_storage, key))
        else:
            from boto.s3.key import Key
            if check_existence and not self.exists(key):
                raise Exception('Can not remove non-existent key %s' % key)
            k = Key(self.bucket)
            k.key = key
            self.bucket.delete_key(k)


def test_storage_local():
    """
    """
    S = Storage(local_storage_folder="~/PVM_test")
    contents = S.list_path("")
    assert contents == []
    S.put(path="", from_path=os.path.abspath(__file__))
    contents = S.list_path("")
    assert len(contents) == 1
    S.remove(__file__)
    contents = S.list_path("")
    assert len(contents) == 0


def test_storage_remote():
    """
    """
    S = Storage(local_storage_folder="~/PVM_test_local_remote", S3_bucket="technical.services.braincorporation.net", S3_credentials=get_S3_credentials())
    contents = S.list_path("")
    N = len(contents)
    S.put(path="", from_path=os.path.abspath(__file__), overwrite=True)
    contents = S.list_path("")
    assert len(contents) == (N + 1)
    S.remove(__file__)
    contents = S.list_path("")
    assert len(contents) == N


if __name__ == "__main__":
    test_storage_local()
    test_storage_remote()

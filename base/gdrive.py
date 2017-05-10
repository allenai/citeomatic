#!/usr/bin/env python
"""
Convenience methods for interacting with the Google Drive API.

In particular, this module provides methods for importing and exporting
Google Sheets documents from a Python script or notebook.
"""

import io
import logging
import os
import os.path
import subprocess

import httplib2
import pandas
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from oauth2client import tools
from oauth2client.file import Storage

# client-secrets from:
S3_CLIENT_SECRETS_JSON = 's3://ai2-secure/google_api_keys/' + \
                         'client-secret-237688346572-p2eajgggc4r3j6qk5vhm42iolrkvfrbo.apps.googleusercontent.com.json'

CONFIG_DIR = os.path.join(os.path.expanduser("~"), '.config', 'google-auth')
LOCAL_CLIENT_SECRETS_JSON = os.path.join(CONFIG_DIR, 'client-secrets.json')

# Metadata fields to request from google drive
DOC_FIELDS = 'id,name,mimeType,webViewLink,parents,kind'

os.system('mkdir -p %s' % CONFIG_DIR)


def fetch_secrets():
    if not os.path.exists(LOCAL_CLIENT_SECRETS_JSON):
        subprocess.check_output(
            [
                'aws', 's3', 'cp', S3_CLIENT_SECRETS_JSON,
                LOCAL_CLIENT_SECRETS_JSON
            ],
            stderr=subprocess.STDOUT
        )
    return LOCAL_CLIENT_SECRETS_JSON


def login(
    service='drive',
    version='v3',
    scope='https://www.googleapis.com/auth/drive'
):
    if not os.path.exists(LOCAL_CLIENT_SECRETS_JSON):
        fetch_secrets()

    from oauth2client.client import flow_from_clientsecrets
    flow = flow_from_clientsecrets(LOCAL_CLIENT_SECRETS_JSON, scope=scope)
    scope_name = scope.replace('/', '_')
    storage = Storage(
        os.path.join(CONFIG_DIR, '%s_credentials.dat' % scope_name)
    )
    credentials = storage.get()

    if credentials is None or credentials.invalid:
        credentials = tools.run_flow(
            flow, storage, tools.argparser.parse_args([])
        )

    http = httplib2.Http()
    http = credentials.authorize(http)
    service = build(service, version, http=http)
    return service


def df_to_excel(df, title):
    import tempfile
    with tempfile.NamedTemporaryFile('wb') as tf:
        writer = pandas.ExcelWriter(tf.name, engine='xlwt')
        df.to_excel(writer, sheet_name=title, index=False)
        writer.close()
        tf.flush()
        return open(tf.name, 'rb').read()


class DriveOps(object):
    def __init__(self, service):
        self.service = service

    def doc_from_name(self, name) -> dict:
        results = self.service.files()\
            .list(q="trashed!=true and name = '%s'" % name).execute()['files']
        if len(results) > 0:
            return results[0]
        return None

    def doc_from_id(self, id) -> dict:
        return self.service.files().get(fileId=id, fields=DOC_FIELDS).execute()

    def copy(self, source_id, dest_name, replace=False):
        if replace is True:
            target_doc = self.doc_from_name(dest_name)
            if target_doc:
                logging.info(
                    'Deleting existing target document: %s', target_doc
                )
                self.service.files().delete(fileId=target_doc['id']).execute()

        source_doc = self.doc_from_id(source_id)
        resp = self.service.files().copy(
            fileId=source_id,
            fields=DOC_FIELDS,
            body={
                'name': dest_name,
                'title': dest_name,
                'parents': source_doc['parents'],
            }
        ).execute()

        return resp


class Document(object):
    """
    Manages a Google drive object.

    A new sheet can be created from a Pandas data frame using `upload`.

    Existing sheets can be retrieved as a DataFrame object and updated with
    a replacement DataFrame.
    """

    def __init__(self, service, file):
        self._service = service
        self._file = file

    def to_df(self):
        content = self._service.files().export(
            fileId=self._file['id'], mimeType='text/csv'
        ).execute()
        return pandas.read_csv(io.StringIO(content.decode('utf8')))

    def to_df_dict(self):
        """Read a dictionary mapping all worksheet names to corresponding dataframes"""
        content = self._service.files().export(
            fileId=self._file['id'],
            mimeType=
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ).execute()
        return pandas.read_excel(io.BytesIO(content), sheetname=None)

    def to_csv(self, filename):
        with open(filename, 'wb') as f:
            content = self._service.files().export(
                fileId=self._file['id'],
                mimeType='text/csv',
            ).execute()
            f.write(content)

    def replace(self, df: pandas.DataFrame):
        """
        Replace the existing sheet with the content of `df`.

        NB: This destroys any existing edits to the sheet.  Use with caution!
        :param df:
        :return:
        """
        fh = io.BytesIO(df_to_excel(df, 'Sheet1'))
        media = MediaIoBaseUpload(
            fh,
            mimetype=
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        self._service.files().update(
            fileId=self._file['id'], media_body=media
        ).execute()

    def delete(self):
        self._service.files().delete(fileId=self._file['id']).execute()

    def __repr__(self):
        return 'Sheet<%s>' % self.url

    @property
    def name(self):
        return self._file['name']

    @property
    def url(self):
        return self._file['webViewLink']

    @property
    def id(self):
        return self._file['id']


def sheet_from_id(service, file_id) -> Document:
    return Document(
        service,
        service.files().get(fileId=file_id, fields=DOC_FIELDS).execute()
    )


def sheet_from_name(service, name) -> Document:
    results = service.files().list(q="name = '%s'" % name).execute()['files']
    if len(results) > 0:
        return Document.from_id(service, results[0]['id'])
    else:
        return None


def sheet_from_upload(
    service,
    data_frame: pandas.DataFrame,
    name: str,
    description: str=None,
    replace: bool=False,
    parent_id=None
) -> Document:
    """
    Upload the given Pandas DataFrame to GDrive and convert it to a Google Sheet.
    """
    if replace:
        existing_sheet = sheet_from_name(service, name)
        if existing_sheet is not None:
            existing_sheet.replace(data_frame)
            return existing_sheet
        logging.info(
            'No existing sheet with name "%s", creating new sheet', name
        )

    fh = io.BytesIO(df_to_excel(data_frame, 'Sheet1'))
    media = MediaIoBaseUpload(
        fh,
        mimetype=
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    body = {
        'name': name,
        'description': description,
        'mimeType': 'application/vnd.google-apps.spreadsheet',
    }
    # Set the parent folder.
    if parent_id:
        body['parents'] = [{'id': parent_id}]

    file = service.files().create(
        fields=DOC_FIELDS, body=body, media_body=media
    ).execute()

    return Document(service, file)


Document.from_upload = sheet_from_upload
Document.from_id = sheet_from_id
Document.from_name = sheet_from_name

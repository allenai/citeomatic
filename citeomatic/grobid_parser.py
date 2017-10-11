import collections
import logging
import re

import arrow
import requests
import untangle
from citeomatic.utils import flatten

date_parser = re.compile(r'[^\d](?:19|20)\d\d[^\d]')
CURRENT_YEAR = arrow.now().year
EARLIEST_YEAR = 1970


def _all_text(doc):
    child_text = [_all_text(c) for c in doc.children]
    cdata_text = doc.cdata.strip() if doc.cdata is not None else ''
    return child_text + [cdata_text]


def _reference_dates(doc):
    if 'date' in [c._name for c in doc.children]:
        try:
            date = int(doc.date['when'])
            return [date]
        except:
            return []
    else:
        return [_reference_dates(c) for c in doc.children]


def _find_latest_year(doc):
    text = ' '.join(flatten(_all_text(doc)))
    ref_dates = flatten(_reference_dates(doc))
    results = [int(r[1:-1]) for r in date_parser.findall(text)] + ref_dates
    results = sorted(results)
    best_result = None
    for r in results:
        if CURRENT_YEAR >= r >= EARLIEST_YEAR:
            best_result = r
    return best_result


def _extract_authors(file_desc):
    try:
        author_groups = file_desc.sourceDesc.biblStruct.analytic.author
    except IndexError:
        logging.warning('Failed to find author group.')
        return []

    authors = []
    for anode in author_groups:
        try:
            forename = anode.persName.forename
            surname = anode.persName.surname

            forename = forename.cdata if hasattr(forename,
                                                 'cdata') else forename[0].cdata
            surname = surname.cdata
            authors.append('%s %s' % (forename, surname))
        except IndexError:
            logging.warning('Failed to parse author %s', anode)
    return authors


def _extract_year(doc, file_desc):
    try:
        return int(file_desc.publicationStmt.date['when'].split("-")[0])
    except Exception as e:
        year_guess = _find_latest_year(doc)
        if year_guess is not None:
            return year_guess
        else:
            return 2017


def _extract_refs(doc):
    try:
        references_list = []
        references = [
            ele for ele in doc.TEI.text.back.div if ele['type'] == 'references'
        ][0]
        for item in references.listBibl.biblStruct:
            references_list.append(item.children[0].title.cdata.lower())
        return references_list
    except IndexError as e:
        logging.warning('Failed to parse references.')
        return []


def _extract_abstract(profile_desc):
    try:
        return profile_desc.abstract.p.cdata
    except IndexError as e:
        logging.warning('Failed to parse abstract', exc_info=1)
        logging.warning('%s', profile_desc)
        raise


def _extract_title(file_desc):
    try:
        return file_desc.titleStmt.title.cdata
    except IndexError as e:
        logging.warning('Failed to parse title', exc_info=1)
        logging.warning('%s', file_desc.titleStmt)
        raise


GrobidResponse = collections.namedtuple(
    'GrobidResponse', ['title', 'authors', 'abstract', 'references', 'year']
)


def parse_full_text(raw) -> GrobidResponse:
    raw = raw
    doc = untangle.parse(raw)
    file_desc = doc.TEI.teiHeader.fileDesc
    profile_desc = doc.TEI.teiHeader.profileDesc

    return GrobidResponse(
        title=_extract_title(file_desc),
        authors=_extract_authors(file_desc),
        abstract=_extract_abstract(profile_desc),
        references=_extract_refs(doc),
        year=_extract_year(doc, file_desc)
    )


def parse_header_text(raw) -> GrobidResponse:
    file_desc = untangle.parse(raw).TEI.teiHeader.fileDesc
    profile_desc = untangle.parse(raw).TEI.teiHeader.profileDesc

    return GrobidResponse(
        title=_extract_title(file_desc),
        authors=_extract_authors(file_desc),
        abstract=_extract_abstract(profile_desc),
        references=[],
        year=2017,
    )


class GrobidParser(object):
    def __init__(self, grobid_url):
        self._grobid_url = grobid_url

    def parse(self, pdf) -> GrobidResponse:
        url = '%s/processFulltextDocument' % self._grobid_url
        xml = requests.post(url, files=[pdf]).text

        try:
            return parse_full_text(xml)
        except:
            logging.warning('Failed to parse full PDF, falling back on header.')
            print('Failed to parse full PDF, falling back on header.')
            url = '%s/processHeaderDocument' % self._grobid_url
            xml = requests.post(url, files=[pdf]).text
            return parse_header_text(xml)

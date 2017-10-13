from citeomatic.common import Document


def document_to_bibtex(doc: Document):
    """
    Return a BibTeX string for the given document.
    :param doc:
    :return: str:
    """
    authors = doc.authors
    if authors:
        author_prefix = authors[0].split(' ')[-1].lower()
    else:
        author_prefix = ''

    title_prefix = doc.title.split()[0].lower()
    params = {
        'title': doc.title,
        'author': ', '.join(doc.authors),
        'venue': doc.venue,
        'year': doc.year,
        'bibname': '%s%s%s' % (author_prefix, doc.year, title_prefix)
    }

    return '''@article{%(bibname)s,
      title={%(title)s},
      author={%(author)s},
      year={%(year)s}
    }''' % params

import logging
import os

from citeomatic import file_util
from citeomatic.common import DatasetPaths, FieldNames
from citeomatic.config import App
from citeomatic.service import document_from_dict, dict_from_document
from citeomatic.traits import Enum
import json


class ConvertKddToCiteomatic(App):
    dataset_name = Enum(options=['dblp', 'pubmed'])

    def main(self, args):

        if self.dataset_name == 'dblp':
            input_path = DatasetPaths.DBLP_GOLD_DIR
            output_path = DatasetPaths.DBLP_CORPUS_JSON
        elif self.dataset_name == 'pubmed':
            input_path = DatasetPaths.PUBMED_GOLD_DIR
            output_path = DatasetPaths.PUBMED_CORPUS_JSON
        else:
            assert False

        logging.info("Reading Gold data from {}".format(input_path))
        logging.info("Writing corpus to {}".format(output_path))
        assert os.path.exists(input_path)
        assert not os.path.exists(output_path)

        papers_file = os.path.join(input_path, "papers.txt")
        abstracts_file = os.path.join(input_path, "abstracts.txt")
        keyphrases_file = os.path.join(input_path, "paper_keyphrases.txt")
        citations_file = os.path.join(input_path, "paper_paper.txt")
        authors_file = os.path.join(input_path, "paper_author.txt")

        venues_file = os.path.join(input_path, "paper_venue.txt")

        paper_titles = {}
        paper_years = {}
        paper_abstracts = {}
        paper_keyphrases = {}
        paper_citations = {}
        paper_in_citations = {}
        paper_authors = {}
        paper_venues = {}

        bad_ids = set()
        for line in file_util.read_lines(abstracts_file):
            parts = line.split("\t")
            paper_id = int(parts[0])
            if len(parts) == 2:
                paper_abstracts[paper_id] = parts[1]
            else:
                paper_abstracts[paper_id] = ""

            if paper_abstracts[paper_id] == "":
                bad_ids.add(paper_id)

        for line in file_util.read_lines(papers_file):
            parts = line.split('\t')
            paper_id = int(parts[0])
            paper_years[paper_id] = int(parts[2])
            paper_titles[paper_id] = parts[3]

        for line in file_util.read_lines(keyphrases_file):
            parts = line.split("\t")
            paper_id = int(parts[0])
            if paper_id not in paper_keyphrases:
                paper_keyphrases[paper_id] = []

            for kp in parts[1:]:
                kp = kp.strip()
                if len(kp) > 0:
                    paper_keyphrases[paper_id].append(kp[:-4])

        for line in file_util.read_lines(citations_file):
            parts = line.split("\t")
            paper_id = int(parts[0])
            if paper_id not in paper_citations:
                paper_citations[paper_id] = []
            c = int(parts[1])
            if c in bad_ids:
                continue
            paper_citations[paper_id].append(str(c))

            if c not in paper_in_citations:
                paper_in_citations[c] = []
            if paper_id not in paper_in_citations:
                paper_in_citations[paper_id] = []

            paper_in_citations[c].append(paper_id)

        for line in file_util.read_lines(authors_file):
            parts = line.split("\t")
            paper_id = int(parts[0])
            if paper_id not in paper_authors:
                paper_authors[paper_id] = []

            paper_authors[paper_id].append(parts[1])

        for line in file_util.read_lines(venues_file):
            parts = line.split("\t")
            paper_id = int(parts[0])
            if paper_id not in paper_venues:
                paper_venues[paper_id] = []

            paper_venues[paper_id].append(parts[1])

        test_paper_id = 13
        print("==== Test Paper Details ====")
        print(paper_titles[test_paper_id])
        print(paper_years[test_paper_id])
        print(paper_abstracts[test_paper_id])
        print(paper_keyphrases[test_paper_id])
        print(paper_citations[test_paper_id])
        print(paper_in_citations[test_paper_id])
        print(paper_authors[test_paper_id])
        print(paper_venues[test_paper_id])
        print("==== Test Paper Details ====")

        def _print_len(x, name=''):
            print("No. of {} = {}".format(name, len(x)))

        _print_len(paper_titles, 'Titles')
        _print_len(paper_years, 'Years')
        _print_len(paper_abstracts, 'Abstracts')
        _print_len(paper_keyphrases, 'KeyPhrases')
        _print_len(paper_citations, 'Paper Citations')
        _print_len(paper_in_citations, 'Paper In citations')
        _print_len(paper_authors, ' Authors')
        _print_len(paper_venues, ' Venues')

        corpus = {}
        for id, title in paper_titles.items():
            if id in bad_ids:
                continue
            doc = document_from_dict(
                {
                    FieldNames.PAPER_ID: str(id),
                    FieldNames.TITLE: title,
                    FieldNames.ABSTRACT: paper_abstracts[id],
                    FieldNames.OUT_CITATIONS: paper_citations.get(id, []),
                    FieldNames.YEAR: paper_years[id],
                    FieldNames.AUTHORS: paper_authors.get(id, []),
                    FieldNames.KEY_PHRASES: paper_keyphrases[id],
                    FieldNames.OUT_CITATIONS_COUNT: len(paper_citations.get(id, [])),
                    FieldNames.IN_CITATIONS_COUNT: len(paper_in_citations.get(id, [])),
                    FieldNames.VENUE: paper_venues.get(id, '')
                }
            )
            corpus[id] = doc

        with open(output_path, 'w') as f:
            for _, doc in corpus.items():
                doc_json = dict_from_document(doc)
                f.write(json.dumps(doc_json))
                f.write("\n")


ConvertKddToCiteomatic.run(__name__)

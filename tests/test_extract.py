import glob

from citeomatic.grobid_parser import GrobidParser, parse_full_text
from citeomatic import file_util


def test_grobid_reed():
    parser = parse_full_text(
        file_util.slurp(file_util.test_file(__file__, 'reed.xml'))
    )
    assert parser.title == 'Optimizing Cauchy Reed-Solomon Codes for Fault-Tolerant Network Storage Applications Optimizing Cauchy Reed-Solomon Codes for Fault-Tolerant Network Storage Applications'
    assert parser.authors == [
        'James Plank', 'Lihao Xu', 'James Plank', 'Lihao Xu'
    ]
    assert parser.abstract == ' '.join(
        [
            'In the past few years, all manner of storage applications , ranging from disk array systems to',
            'distributed and wide-area systems, have started to grapple with the reality of tolerating multiple',
            'simultaneous failures of storage nodes. Unlike the single failure case, which is optimally handled',
            'with RAID Level-5 parity, the multiple failure case is more difficult because optimal general purpose',
            'strategies are not yet known. Erasure Coding is the field of research that deals with these',
            'strategies, and this field has blossomed in recent years. Despite this research, the decades-old',
            'Reed-Solomon erasure code remains the only space-optimal (MDS) code for all but the smallest storage',
            'systems. The best performing implementations of Reed-Solomon coding employ a variant called Cauchy',
            "Reed-Solomon coding, developed in the mid 1990's [4]. In this paper, we present an improvement to",
            'Cauchy Reed-Solomon coding that is based on optimizing the Cauchy distribution matrix. We detail',
            'an algorithm for generating good matrices and then evaluate the performance of encoding using all',
            'implementations Reed-Solomon codes, plus the best MDS codes from the literature. The improvements',
            'over the original Cauchy Reed-Solomon codes are as much as 83% in realistic scenarios, and average',
            'roughly 10% over all cases that we tested.'
        ]
    )


def test_grobid_salience():
    parser = parse_full_text(
        file_util.slurp(file_util.test_file(__file__, 'salience.xml'))
    )
    assert parser.title == 'A Model of Saliency-based Visual Attention for Rapid Scene Analysis'
    assert parser.authors == ['Laurent Itti', 'Christof Koch', 'Ernst Niebur']
    assert parser.abstract == ' '.join(
        [
            '{ A visual attention system, inspired by the behavior and the neuronal',
            'architecture of the early primate visual system, is presented.',
            'Multiscale image features are combined into a single topographical',
            'saliency map. A dynamical neu-ral network then selects attended',
            'locations in order of decreasing saliency. The system breaks down the',
            'complex problem of scene understanding by rapidly selecting, in a',
            'computationally eecient manner, conspicuous locations to be analyzed',
            'in detail.'
        ]
    )


def _test_all():
    for pdf in glob.glob('./data/pdfs/*.pdf'):
        pdf_blob = ('input', ('pdf', open(pdf, 'rb').read(), 'application/pdf'))
        try:
            parsed = GrobidParser('http://localhost:8080').parse(pdf_blob)
            print(pdf, parsed.title, parsed.authors)
        except:
            print('Failed to parse: %s', pdf)


if __name__ == '__main__':
    import pytest

    pytest.main(['-s', __file__])

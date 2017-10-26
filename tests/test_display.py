import re

from citeomatic.common import Document, global_tokenizer
from citeomatic import display

TEST_ABSTRACT = """
'— This paper investigates into the colorization problem which converts a grayscale image to a colorful version. This is a very difficult problem and normally requires manual adjustment to achieve artifact-free quality. For instance , it normally requires human-labelled color scribbles on the grayscale target image or a careful selection of colorful reference images (e.g., capturing the same scene in the grayscale target image). Unlike the previous methods, this paper aims at a high-quality fully-automatic colorization method. With the assumption of a perfect patch matching technique, the use of an extremely large-scale reference database (that contains sufficient color images) is the most reliable solution to the colorization problem. However, patch matching noise will increase with respect to the size of the reference database in practice. Inspired by the recent success in deep learning techniques which provide amazing modeling of large-scale data, this paper re-formulates the colorization problem so that deep learning techniques can be directly employed. To ensure artifact-free quality, a joint bilateral filtering based post-processing step is proposed. We further develop an adaptive image clustering technique to incorporate the global image information. Numerous experiments demonstrate that our method outperforms the state-of-art algorithms both in terms of quality and speed.'"""

TEST_DOCS = [
    Document(
        title=' '.join(global_tokenizer('Deep Colorization')),
        title_raw='Deep Colorization',
        abstract=' '.join(global_tokenizer(TEST_ABSTRACT)),
        abstract_raw=TEST_ABSTRACT,
        authors=['Zezhou Cheng', 'Qingxiong Yang', 'Bin Sheng'],
        out_citations=[],
        year=2015,
        id='6baaca1b6de31ac2a5b1f89e9b3baa61e41d52f9',
        venue='ICCV',
        in_citation_count=8,
        out_citation_count=37,
        key_phrases=[
            'Colorization', 'Reference Database', 'Deep Learning Technique'
        ]
    ), Document(
        title=' '.join(global_tokenizer('Deep Computing')),
        title_raw='Deep Computing',
        abstract='',
        abstract_raw='',
        authors=['Louis V. Gerstner'],
        out_citations=[],
        year=2000,
        id='100544cf556dd8d98e6871bf28ea9e87a6f0ecc9',
        venue='LOG IN',
        in_citation_count=0,
        out_citation_count=0,
        key_phrases=[]
    ), Document(
        title=' '.join(global_tokenizer('Deep Blue')),
        title_raw='Deep Blue',
        abstract='',
        abstract_raw='',
        authors=['Jim Ottaviani'],
        out_citations=[],
        year=2006,
        id='60a5511f544d0ed5155f7c5f0a70b8c87337d2f7',
        venue='IASSIST Conference',
        in_citation_count=0,
        out_citation_count=0,
        key_phrases=[]
    )
]

EXPECTED_BIBTEX = [
    """@article{cheng2015deep,
          title={Deep Colorization},
          author={Zezhou Cheng, Qingxiong Yang, Bin Sheng},
          year={2015}
        }""", """@article{gerstner2000deep,
          title={Deep Computing},
          author={Louis V. Gerstner},
          year={2000}
        }""", """@article{ottaviani2006deep,
          title={Deep Blue},
          author={Jim Ottaviani},
          year={2006}
        }"""
]


def test_bibtex_export():
    for doc, expected in zip(TEST_DOCS, EXPECTED_BIBTEX):
        assert re.sub('\\s+', ' ', display.document_to_bibtex(doc)).lower() == \
               re.sub('\\s+', ' ', expected).lower()

import stanza
from stanza import DownloadMethod
from stanza.models.common.doc import Sentence
from typing import List

stanza.download("en")
# set tokenize_pretokenized=True to prevent further tokenization or sentence segmentation
# see more: https://stanfordnlp.github.io/stanza/tokenize.html#start-with-pretokenized-text
nlp = stanza.Pipeline("en",
                      processors='tokenize,pos,depparse,sentiment,mwt,lemma',
                      tokenize_pretokenized=True,
                      download_method=DownloadMethod.REUSE_RESOURCES,
                      tokenize_no_ssplit=True)

__all__ = ['annotation', 'annotation_plus']

def annotation(text: str) -> Sentence:
    """sentence annotation

    Parameters
    ----------
    text : str
        sentence needed to be annotated

    Returns
    -------
    List[dict]
        dict information about the annotated sentence. 
        see more: https://stanfordnlp.github.io/stanza/data_conversion.html#document-to-python-object
    """
    doc = nlp(text)
    sentences = doc.sentences
    assert len(sentences) == 1, "the text must be a sentence. Got {} instead".format(text)
    return doc.sentences[0]


def annotation_plus(documents: List[str]) -> List[Sentence]:
    docs = nlp([stanza.Document([], text=d) for d in documents])
    return [doc.sentences[0] for doc in docs]
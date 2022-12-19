import os
from pathlib import Path


class Constants:
    DATA_ROOT: str = os.path.join(
        os.path.expanduser("~"), ".cache", "forecite"
    )

    # Related to the output format of the citation scores
    TERM_OCCURRENCES_INDEX: int = 2
    TERM_CITATIONS_INDEX: int = 1

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

        Path(self.DATA_ROOT).mkdir(parents=True, exist_ok=True)

    @property
    def TOPICID_DATA_ROOT(self):
        return os.path.join(self.DATA_ROOT, "topic_identification")

    @property
    def NO_REFS_ARXIV_CS_DATA_ROOT(self):
        return os.path.join(self.DATA_ROOT, "arxiv_no_refs")

    # Processed data file paths
    @property
    def NO_REFS_ARXIV_CS_IDS_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "arxiv_ids.json"
        )

    @property
    def NO_REFS_ARXIV_CS_TO_S2_MAPPING_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "arxiv_to_s2_mapping.json"
        )

    @property
    def NO_REFS_ARXIV_CS_TITLE_NPS_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "title_nps.json"
        )

    @property
    def NO_REFS_ARXIV_CS_ABSTRACT_NPS_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "abstract_nps.json"
        )

    @property
    def NO_REFS_ARXIV_CS_BODY_NPS_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "body_nps.json"
        )

    @property
    def NO_REFS_ARXIV_CS_NORMALIZATION_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "normalization.json"
        )

    @property
    def NO_REFS_ARXIV_CS_REFERENCES_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "s2_id_to_references.json"
        )

    @property
    def NO_REFS_ARXIV_CS_CITING_IDS_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "s2_id_to_citing_ids.json"
        )

    @property
    def NO_REFS_ARXIV_CS_CANONICALIZATION_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "s2_id_to_canonical.json"
        )

    @property
    def NO_REFS_ARXIV_CS_TITLE_CANDIDATES_SCORES_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "title_citation_scores.json"
        )

    @property
    def NO_REFS_ARXIV_CS_TITLE_CANDIDATES_CNLC_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "title_cnlc_scores.json"
        )

    @property
    def NO_REFS_ARXIV_CS_TITLE_CANDIDATES_LOOR_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "title_loor_scores.json"
        )

    @property
    def NO_REFS_ARXIV_CS_ABSTRACT_CANDIDATES_SCORES_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "abstract_citation_scores.json"
        )

    @property
    def NO_REFS_ARXIV_CS_ABSTRACT_CANDIDATES_CNLC_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "abstract_cnlc_scores.json"
        )

    @property
    def NO_REFS_ARXIV_CS_ABSTRACT_CANDIDATES_LOOR_PATH(self):
        return os.path.join(
            self.NO_REFS_ARXIV_CS_DATA_ROOT, "abstract_loor_scores.json"
        )

    # Evaluation related
    @property
    def RANDOM_SAMPLE_FOR_CALIBRATION_OUTPUT_PATH(self):
        return os.path.join(
            self.TOPICID_DATA_ROOT, "random_noun_phrase_calibration_sample.csv"
        )

    @property
    def RANDOM_SAMPLE_FOR_EVALUATION_OUTPUT_PATH(self):
        return os.path.join(
            self.TOPICID_DATA_ROOT, "random_noun_phrase_sample.csv"
        )

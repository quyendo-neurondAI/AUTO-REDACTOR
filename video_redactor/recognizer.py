import os
import re
import logging
from typing import List, Optional
from presidio_analyzer import EntityRecognizer, RecognizerResult, AnalyzerEngine
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pathlib import Path

ENTITIES_LIST = [
    'PASSWORD', 'ZIPCODE', 'AMBIGUOUS', 'BUILDINGNUM', 'CREDITCARDNUMBER',
    'USERNAME', 'CITY', 'O', 'KEY', 'STREET', 'ACCOUNTNUM', 'TELEPHONENUM',
    'DATEOFBIRTH', 'IP_ADDRESS', 'CRYPTO', 'CREDIT_CARD', 'NAME', 'EMAIL'
]

class HuggingFacePIIRecognizer(EntityRecognizer):
    def __init__(self, model_path: str, supported_entities: Optional[List[str]] = None):
        normalized_entities = list({self._normalize_entity(ent) for ent in (supported_entities or [])})
        super().__init__(supported_entities=normalized_entities)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_path, device_map="cpu", local_files_only=True
        )
        self.nlp_pipeline = pipeline(
            "token-classification", model=self.model, tokenizer=self.tokenizer, aggregation_strategy="simple"
        )

        if not supported_entities:
            raw_labels = list(set(self.model.config.id2label.values()))
            self.supported_entities = list({self._normalize_entity(label) for label in raw_labels})

        self.name = model_path.split("/")[-1] + "_recognizer"
        self.logger = logging.getLogger(__name__)

    def load(self) -> bool:
        return True

    def analyze(self, text: str, entities: List[str] = None, nlp_artifacts=None) -> List[RecognizerResult]:
        entities_to_detect = {self._normalize_entity(e) for e in (entities or self.supported_entities)}
        results = []
        predictions = self.nlp_pipeline(text)

        for pred in predictions:
            entity_type = self._normalize_entity(pred["entity_group"])
            if entity_type not in entities_to_detect:
                continue
            results.append(RecognizerResult(
                entity_type=entity_type, start=pred["start"], end=pred["end"],
                score=pred["score"], analysis_explanation=None
            ))
        return results

    @staticmethod
    def _normalize_entity(label: str) -> str:
        return re.sub(r"^(B-|I-)", "", label)

def build_analyzer():
    analyzer = AnalyzerEngine()
    base_path = Path(__file__).parent.resolve()
    recognizers = [
        HuggingFacePIIRecognizer(model_path=os.path.join(base_path, r"models\starpii")),
        HuggingFacePIIRecognizer(model_path=os.path.join(base_path, r"models\piiranha"))
    ]
    for rec in recognizers:
        analyzer.registry.add_recognizer(rec)
    return analyzer

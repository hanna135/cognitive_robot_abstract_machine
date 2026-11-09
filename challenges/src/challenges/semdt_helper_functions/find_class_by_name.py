from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from krrood.utils import recursive_subclasses

# Imported so that every semantic annotation class is defined before the lookup.
import semantic_digital_twin.semantic_annotations.semantic_annotations  # noqa: F401
from challenges.semdt_helper_functions.exceptions import (
    AmbiguousSemanticAnnotationName,
    UnknownSemanticAnnotationName,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


# %% Name lookup


@dataclass
class SemanticAnnotationClassFinder:
    """
    Finds the semantic annotation class a spoken or written object name refers to, such as
    ``"cola"`` for a class ``Cola`` or ``"pringles can"`` for ``PringlesCan``.

    Names match regardless of case and of spaces, underscores and hyphens between words.
    """

    word_separators: str = " _-"
    """
    The characters that separate the words of a name and are ignored when matching.
    """

    def find(self, name: str) -> type[SemanticAnnotation]:
        """
        :param name: The name of an object, such as ``"cola"`` or ``"pringles_can"``.
        :return: The semantic annotation class with that name.
        :raises UnknownSemanticAnnotationName: If no class has that name.
        :raises AmbiguousSemanticAnnotationName: If several classes have that name.
        """
        normalized_name = self.normalize(name)
        matches = {
            annotation_class
            for annotation_class in recursive_subclasses(SemanticAnnotation)
            if self.normalize(annotation_class.__name__) == normalized_name
        }
        if not matches:
            raise UnknownSemanticAnnotationName(name=name)
        if len(matches) > 1:
            raise AmbiguousSemanticAnnotationName(name=name, matches=list(matches))
        return matches.pop()

    def normalize(self, name: str) -> str:
        """
        :param name: A class name or object name.
        :return: The name in lower case without word separators.
        """
        return name.translate(str.maketrans("", "", self.word_separators)).lower()

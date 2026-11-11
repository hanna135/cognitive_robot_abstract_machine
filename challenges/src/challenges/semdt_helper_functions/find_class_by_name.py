from __future__ import annotations

# %% Imports
import inspect
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
    Finds the semantic annotation class a spoken or written object name refers to, such
    as ``"cola"`` for a class ``Cola`` or ``"pringles can"`` for ``PringlesCan``.

    Names match regardless of case and of spaces, underscores and hyphens between words.
    A name no class has exactly falls back to the concrete classes whose name contains
    it, such as ``"counter"`` for ``CounterTop``.
    """

    word_separators: str = " _-"
    """
    The characters that separate the words of a name and are ignored when matching.
    """

    def find(self, name: str) -> type[SemanticAnnotation]:
        """
        :param name: The name of an object, such as ``"cola"`` or ``"pringles_can"``.
        :return: The semantic annotation class with that name, or else the one concrete
            class whose name contains it.
        :raises UnknownSemanticAnnotationName: If no class has or contains that name.
        :raises AmbiguousSemanticAnnotationName: If several classes have that name, or
            none has it and several concrete classes contain it.
        """
        normalized_name = self.normalize(name)
        annotation_classes = recursive_subclasses(SemanticAnnotation)
        matches = {
            annotation_class
            for annotation_class in annotation_classes
            if self.normalize(annotation_class.__name__) == normalized_name
        } or {
            annotation_class
            for annotation_class in annotation_classes
            if normalized_name in self.normalize(annotation_class.__name__)
            and not inspect.isabstract(annotation_class)
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

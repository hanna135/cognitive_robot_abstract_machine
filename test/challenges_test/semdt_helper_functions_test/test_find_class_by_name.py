from __future__ import annotations

# %% Imports
from dataclasses import dataclass

import pytest

from challenges.semdt_helper_functions.exceptions import (
    AmbiguousSemanticAnnotationName,
    UnknownSemanticAnnotationName,
)
from challenges.semdt_helper_functions.find_class_by_name import (
    SemanticAnnotationClassFinder,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    DrinkingContainer,
    Milk,
)
from semantic_digital_twin.world_description.world_entity import SemanticAnnotation


# %% Fixtures


@pytest.fixture
def finder() -> SemanticAnnotationClassFinder:
    """
    A finder with the default word separators.
    """
    return SemanticAnnotationClassFinder()


# %% Matching


def test_lower_case_name_finds_class(finder: SemanticAnnotationClassFinder):
    assert finder.find("milk") is Milk


@pytest.mark.parametrize(
    "name", ["drinking container", "drinking_container", "Drinking-Container"]
)
def test_name_with_word_separators_finds_class(
    finder: SemanticAnnotationClassFinder, name: str
):
    assert finder.find(name) is DrinkingContainer


# %% Failures


def test_unknown_name_raises(finder: SemanticAnnotationClassFinder):
    with pytest.raises(UnknownSemanticAnnotationName):
        finder.find("no such object")


def test_name_shared_by_two_classes_raises(finder: SemanticAnnotationClassFinder):
    @dataclass(eq=False)
    class SharedName(SemanticAnnotation): ...

    first_class = SharedName

    @dataclass(eq=False)
    class SharedName(SemanticAnnotation): ...

    with pytest.raises(AmbiguousSemanticAnnotationName) as raised:
        finder.find("shared name")
    assert set(raised.value.matches) == {first_class, SharedName}

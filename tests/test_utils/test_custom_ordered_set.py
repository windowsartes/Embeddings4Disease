import typing as tp

import pytest

from embeddings4disease.utils.utils import CustomOrderedSet


test_data = [
    [1,],
    [1, 2],
    [1, 2, 3],
    [1, 2, 3, 4],
]

@pytest.mark.parametrize("numbers", test_data)
def test_all_uniques(numbers: list[int]):
    custom_ordered_set: CustomOrderedSet = CustomOrderedSet()

    for number in numbers:
        custom_ordered_set.add(number)

    assert set(numbers) == set([number for number in custom_ordered_set])


def test_empty_set():
    custom_ordered_set: CustomOrderedSet = CustomOrderedSet()

    assert set() == set([value for value in custom_ordered_set])


test_data = [
    [1,],
    [2, 2],
    [3, 3, 3],
    [4, 4, 4, 4],
    [5, 5, 5, 5, 5],
]

@pytest.mark.parametrize("numbers", test_data)
def test_all_same(numbers: list[int]):
    custom_ordered_set: CustomOrderedSet = CustomOrderedSet()

    for number in numbers:
        custom_ordered_set.add(number)

    assert len([number for number in custom_ordered_set]) == 1
    assert numbers[0] == [number for number in custom_ordered_set][0]


test_data = [
    [0],
    [1, 2],
    [3, 2, 1],
    [4, 6, 5, 7],
]

@pytest.mark.parametrize("numbers", test_data)
def test_write_order(numbers: list[int]):
    custom_ordered_set: CustomOrderedSet = CustomOrderedSet()

    for number in numbers:
        custom_ordered_set.add(number)

    assert numbers == [value for value in custom_ordered_set]


test_data = [
    ([1,], [1,]),
    ([2, 2,], [2,]),
    ([3, 3, 3,], [3,]),
    ([4, 4, 5, 5], [4, 5]),
    ([6, 7, 7, 8, 8, 8], [6, 7, 8]),
]

@pytest.mark.parametrize("input_lst,answer_lst", test_data)
def test_remove_duplicates(input_lst: list[int], answer_lst: list[int]):
    custom_ordered_set: CustomOrderedSet = CustomOrderedSet()

    for number in input_lst:
        custom_ordered_set.add(number)

    assert answer_lst == [value for value in custom_ordered_set]


T = tp.TypeVar("T", bound=object)

test_data = [
    ([1, "two"], [1, "two"]),
    ([3, "three", 3.1,], [3, "three", 3.1]),
    ([4, "four", 4, "four"], [4, "four"]),
    (["five", [6, 7, 8.]], ["five", [6, 7, 8.]])
]

@pytest.mark.parametrize("input_lst,answer_lst", test_data)
def test_different_input_types(input_lst: list[T], answer_lst: list[T]):
    custom_ordered_set: CustomOrderedSet = CustomOrderedSet()

    for number in input_lst:
        custom_ordered_set.add(number)

    assert answer_lst == [value for value in custom_ordered_set]

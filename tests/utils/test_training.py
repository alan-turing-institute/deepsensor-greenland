import pytest
import torch

from ds_greenland.utils.training import batched, task_collate_fn


class TestBatched:
    def test_splits_into_chunks_of_n(self):
        result = list(batched("ABCDEFG", 3))
        assert result == [("A", "B", "C"), ("D", "E", "F"), ("G",)]

    def test_exact_multiple_has_no_remainder_chunk(self):
        result = list(batched(range(6), 2))
        assert result == [(0, 1), (2, 3), (4, 5)]

    def test_empty_iterable_yields_nothing(self):
        assert list(batched([], 3)) == []

    def test_n_larger_than_iterable_yields_single_chunk(self):
        assert list(batched([1, 2], 10)) == [(1, 2)]

    def test_raises_for_non_positive_n(self):
        with pytest.raises(ValueError, match="n must be at least one"):
            list(batched([1, 2, 3], 0))


class TestTaskCollateFn:
    def test_returns_expected_keys(self, tiny_dp_and_task_loader):
        _, task_loader, dates = tiny_dp_and_task_loader
        task = task_loader(dates[0], context_sampling="all", target_sampling="all")

        result = task_collate_fn([task])

        assert result.keys() == {"context_data", "xt", "yt", "model_kwargs"}
        assert isinstance(result["yt"], torch.Tensor)

    def test_batch_dimension_matches_number_of_tasks(self, tiny_dp_and_task_loader):
        _, task_loader, dates = tiny_dp_and_task_loader
        tasks = [
            task_loader(date, context_sampling="all", target_sampling="all")
            for date in dates
        ]

        result_single = task_collate_fn(tasks[:1])
        result_all = task_collate_fn(tasks)

        assert result_single["yt"].shape[0] == 1
        assert result_all["yt"].shape[0] == len(dates)
        assert result_single["yt"].shape[1:] == result_all["yt"].shape[1:]

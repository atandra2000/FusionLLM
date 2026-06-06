"""Unit tests for batch-size and seq-len schedules (Phase 4.3)."""

from __future__ import annotations

import pytest

from training.schedules import BatchSizeSchedule, SeqLenSchedule


class TestBatchSizeSchedule:
    def test_initial_value(self):
        s = BatchSizeSchedule(initial_batch_size=2, final_batch_size=8, schedule_steps=100)
        assert s.get_batch_size(0) == 2

    def test_final_value(self):
        s = BatchSizeSchedule(initial_batch_size=2, final_batch_size=8, schedule_steps=100)
        assert s.get_batch_size(100) == 8
        assert s.get_batch_size(200) == 8

    def test_interpolated_value(self):
        s = BatchSizeSchedule(initial_batch_size=2, final_batch_size=8, schedule_steps=100)
        mid = s.get_batch_size(50)
        assert 2 <= mid <= 8

    def test_step_shape_jumps_at_midpoint(self):
        s = BatchSizeSchedule(initial_batch_size=2, final_batch_size=8, schedule_steps=100, shape="step")
        assert s.get_batch_size(49) == 2
        assert s.get_batch_size(50) == 8

    def test_monotonic_increase(self):
        s = BatchSizeSchedule(initial_batch_size=2, final_batch_size=8, schedule_steps=1000)
        values = [s.get_batch_size(i) for i in range(0, 1001, 10)]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1]

    def test_initial_equals_final(self):
        s = BatchSizeSchedule(initial_batch_size=4, final_batch_size=4, schedule_steps=100)
        for step in [0, 50, 100, 200]:
            assert s.get_batch_size(step) == 4

    def test_invalid_params_raises(self):
        with pytest.raises(AssertionError):
            BatchSizeSchedule(initial_batch_size=8, final_batch_size=2)


class TestSeqLenSchedule:
    def test_initial_value(self):
        s = SeqLenSchedule(initial_seq_len=2048, final_seq_len=8192, schedule_steps=100)
        assert s.get_seq_len(0) == 2048

    def test_final_value(self):
        s = SeqLenSchedule(initial_seq_len=2048, final_seq_len=8192, schedule_steps=100)
        assert s.get_seq_len(100) == 8192

    def test_interpolated_value(self):
        s = SeqLenSchedule(initial_seq_len=2048, final_seq_len=8192, schedule_steps=100)
        mid = s.get_seq_len(50)
        assert 2048 <= mid <= 8192

    def test_step_shape_jumps(self):
        s = SeqLenSchedule(initial_seq_len=2048, final_seq_len=8192, schedule_steps=100, shape="step")
        assert s.get_seq_len(49) == 2048
        assert s.get_seq_len(50) == 8192

    def test_initial_equals_final(self):
        s = SeqLenSchedule(initial_seq_len=4096, final_seq_len=4096, schedule_steps=100)
        for step in [0, 50, 100]:
            assert s.get_seq_len(step) == 4096

    def test_invalid_params_raises(self):
        with pytest.raises(AssertionError):
            SeqLenSchedule(initial_seq_len=8192, final_seq_len=2048)

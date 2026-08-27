from pathlib import Path

from chordscope.analysis.separation import SourceSeparator


def test_demucs_local_repo_requires_yaml_and_four_checkpoints(tmp_path: Path):
    assert not SourceSeparator._local_repo_ready(tmp_path)
    for name in SourceSeparator._LOCAL_FILES:
        (tmp_path / name).write_bytes(b"x")
    assert SourceSeparator._local_repo_ready(tmp_path)

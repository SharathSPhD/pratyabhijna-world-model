"""Tests for the Skill Library (SQLite + FAISS retrieval)."""

import shutil
import tempfile
import pytest
from pwm.memory.skill_lib import Skill, SkillLibrary


@pytest.fixture
def tmp_lib(tmp_path):
    """Fresh SkillLibrary in a temp directory."""
    lib = SkillLibrary(db_dir=tmp_path / "skill_lib", quality_threshold=0.5)
    yield lib
    lib.close()


class TestSkillLibraryBasic:
    def test_empty_on_creation(self, tmp_lib):
        assert len(tmp_lib) == 0

    def test_add_and_count(self, tmp_lib):
        added = tmp_lib.add_skill("test_skill", "A creative metaphor skill", camatk_score=0.8)
        assert added is True
        assert len(tmp_lib) == 1

    def test_quality_threshold_enforced(self, tmp_lib):
        added = tmp_lib.add_skill("low_quality", "Low quality skill", camatk_score=0.3)
        assert added is False
        assert len(tmp_lib) == 0

    def test_force_bypasses_threshold(self, tmp_lib):
        added = tmp_lib.add_skill("forced", "Forced low quality skill", camatk_score=0.3, force=True)
        assert added is True
        assert len(tmp_lib) == 1

    def test_duplicate_skill_updates(self, tmp_lib):
        tmp_lib.add_skill("dup", "Version 1", camatk_score=0.7)
        tmp_lib.add_skill("dup", "Version 2", camatk_score=0.9)
        assert len(tmp_lib) == 1   # no duplicate row
        skills = tmp_lib.list_skills()
        assert skills[0].camatk_score == pytest.approx(0.9, abs=0.01)

    def test_list_skills_ordered_by_score(self, tmp_lib):
        tmp_lib.add_skill("low", "Low", camatk_score=0.6)
        tmp_lib.add_skill("high", "High", camatk_score=0.95)
        tmp_lib.add_skill("mid", "Mid", camatk_score=0.75)
        skills = tmp_lib.list_skills()
        scores = [s.camatk_score for s in skills]
        assert scores == sorted(scores, reverse=True)


class TestSkillRetrieval:
    def test_retrieve_returns_skills(self, tmp_lib):
        tmp_lib.add_skill("s1", "Creative metaphor for nature imagery", camatk_score=0.8)
        tmp_lib.add_skill("s2", "Rhythmic Sanskrit metre detection", camatk_score=0.9)
        results = tmp_lib.retrieve("nature and imagery", k=2)
        assert len(results) >= 1
        assert all(isinstance(s, Skill) for s in results)

    def test_retrieve_empty_library(self, tmp_lib):
        results = tmp_lib.retrieve("anything", k=5)
        assert results == []

    def test_retrieve_k_limits_results(self, tmp_lib):
        for i in range(10):
            tmp_lib.add_skill(f"skill_{i}", f"Skill {i} description", camatk_score=0.7 + i * 0.02)
        results = tmp_lib.retrieve("generic query", k=3)
        assert len(results) <= 3


class TestSkillLibraryPersistence:
    def test_skills_persist_across_reload(self, tmp_path):
        lib1 = SkillLibrary(db_dir=tmp_path / "skill_lib")
        lib1.add_skill("persist_test", "This should persist", camatk_score=0.85)
        lib1.close()

        lib2 = SkillLibrary(db_dir=tmp_path / "skill_lib")
        assert len(lib2) == 1
        skills = lib2.list_skills()
        assert skills[0].name == "persist_test"
        lib2.close()

    def test_delete_skill(self, tmp_lib):
        tmp_lib.add_skill("to_delete", "Temporary skill", camatk_score=0.8)
        assert len(tmp_lib) == 1
        deleted = tmp_lib.delete_skill("to_delete")
        assert deleted is True
        assert len(tmp_lib) == 0

    def test_delete_nonexistent_returns_false(self, tmp_lib):
        assert tmp_lib.delete_skill("does_not_exist") is False

    def test_export_json(self, tmp_lib, tmp_path):
        import json
        tmp_lib.add_skill("export_skill", "Exported skill", camatk_score=0.75)
        out_path = tmp_path / "export.json"
        tmp_lib.export_json(out_path)
        data = json.loads(out_path.read_text())
        assert len(data) == 1
        assert data[0]["name"] == "export_skill"

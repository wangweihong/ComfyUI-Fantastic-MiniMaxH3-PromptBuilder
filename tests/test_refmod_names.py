"""RefMod name regressions; run with python -m unittest discover -s tests -v."""
import importlib.util
import inspect
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "_refmod_name_tests"


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


refmods = load_module(PACKAGE + ".refmods", "refmods.py")


def load_create_module():
    # Validation must run on CPU-only hosts without importing ComfyUI or
    # registering its HTTP routes. Only unused encoding dependencies are stubbed;
    # the naming helpers and complete Create node are the real production code.
    modules = {}
    for name in (PACKAGE, "torch", "torch.nn", "torch.nn.functional",
                 "safetensors", "safetensors.torch", "comfy", "comfy.utils",
                 "comfy.model_management", PACKAGE + ".media_io",
                 PACKAGE + ".object_mask", PACKAGE + ".refmod_core",
                 PACKAGE + ".video_edit"):
        modules[name] = types.ModuleType(name)
    modules[PACKAGE].__path__ = [str(ROOT)]
    modules[PACKAGE + ".refmods"] = refmods
    modules["torch"].nn = modules["torch.nn"]
    modules["torch.nn"].functional = modules["torch.nn.functional"]
    modules["safetensors.torch"].save_file = Mock()
    modules[PACKAGE + ".object_mask"].drop_subject_masks = Mock()
    modules[PACKAGE + ".object_mask"].subject_mask = Mock()
    modules[PACKAGE + ".refmod_core"].H3RefMod = Mock()
    modules[PACKAGE + ".refmod_core"].encoder_record = Mock()
    modules[PACKAGE + ".video_edit"].blur_outside = Mock()
    with patch.dict(sys.modules, modules):
        return load_module(PACKAGE + ".refmod_create", "refmod_create.py")


create = load_create_module()
Node = create.MiniMaxH3FantasticRefModCreate


class RefModNamesTest(unittest.TestCase):
    def test_unicode_names_survive_sanitizing_and_resolution(self):
        for name in ("天使", "hd8_天使", "角色2", "Alice-01", "café", "人物_甲"):
            with self.subTest(name=name):
                self.assertEqual(refmods.sanitize_name(name), name)
                self.assertTrue(refmods.valid_rel("角色/" + name))
                self.assertIs(Node.VALIDATE_INPUTS(name=name, subfolder="角色"), True)

    def test_ascii_filename_cleanup_is_preserved(self):
        self.assertEqual(refmods.sanitize_name(" hero:*? "), "hero")
        self.assertEqual(refmods.sanitize_name("characters/hero"), "characters_hero")
        self.assertEqual(refmods.sanitize_name("characters/hero", folder=True), "characters/hero")

    def test_unsafe_relative_paths_remain_invalid(self):
        for name in ("", "../天使", "/天使", "C:/天使", "角色/../天使",
                     "角色/./天使", "角色//天使", ".git/天使", "graph_presets/天使",
                     "角色\x00", "角色\n名字", "角色:名字", "../actor1"):
            with self.subTest(name=name):
                self.assertFalse(refmods.valid_rel(name))

    def test_existing_unicode_name_is_still_taken(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / "角色"
            folder.mkdir()
            (folder / "天使.safetensors").touch()
            with patch.object(refmods, "search_dirs", return_value=[directory]):
                self.assertTrue(refmods.name_taken("角色/天使"))
                self.assertFalse(refmods.name_taken("角色/其他角色"))

    def test_unicode_symlink_escape_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "models"
            outside = Path(directory) / "outside"
            root.mkdir()
            outside.mkdir()
            try:
                (root / "角色").symlink_to(outside, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            with self.assertRaisesRegex(ValueError, "escapes"):
                refmods._contained_target(str(root), "角色/天使")


class CreateValidationTest(unittest.TestCase):
    def test_linked_name_can_be_absent_during_preflight(self):
        self.assertIs(Node.VALIDATE_INPUTS(subfolder="characters", subject_name="Character"), True)

    def test_linked_folder_can_be_absent_during_preflight(self):
        self.assertIs(Node.VALIDATE_INPUTS(name="天使"), True)

    def test_literal_empty_name_is_still_rejected(self):
        for name in ("", " ", "...", "___"):
            with self.subTest(name=name):
                self.assertEqual(Node.VALIDATE_INPUTS(name=name), "Give the RefMod a name.")

    def test_metadata_is_checked_even_when_name_is_linked(self):
        self.assertIsNot(Node.VALIDATE_INPUTS(subject_name="two words"), True)
        self.assertIsNot(Node.VALIDATE_INPUTS(appearance="a" * 301), True)
        self.assertIsNot(Node.VALIDATE_INPUTS(voice_description="a" * 301), True)
        self.assertIsNot(Node.VALIDATE_INPUTS(retained_attributes="a" * 301), True)

    def test_numeric_fields_are_left_to_comfyui_validation(self):
        parameters = inspect.signature(Node.VALIDATE_INPUTS).parameters
        self.assertNotIn("ref_resolution", parameters)
        self.assertNotIn("audio_max_seconds", parameters)
        self.assertFalse(any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()))

    def test_resolved_invalid_values_fail_before_media_loading(self):
        inputs = dict(name="天使", subfolder="characters", mode="Full Reference",
                      ref_resolution=1024, grid=16, latent_frames=22, refinement_steps=0,
                      max_tokens=16384, audio_max_seconds=30.0, concept_type="identity",
                      description="", write_preview=True, source="[]")
        for values, error in (({"name": ""}, "Give the RefMod a name"),
                              ({"name": None}, "Give the RefMod a name"),
                              ({"subject_name": "two words"}, "subject name")):
            with self.subTest(values=values), patch.object(create, "parse_sources") as load:
                with self.assertRaisesRegex(ValueError, error):
                    Node().create(**(inputs | values))
                load.assert_not_called()

    def test_resolved_unicode_name_reaches_media_loading(self):
        inputs = dict(name="天使", subfolder="角色", mode="Full Reference",
                      ref_resolution=1024, grid=16, latent_frames=22, refinement_steps=0,
                      max_tokens=16384, audio_max_seconds=30.0, concept_type="identity",
                      description="", write_preview=True, source="[]")
        with patch.object(create, "parse_sources", side_effect=RuntimeError("media reached")) as load:
            with self.assertRaisesRegex(RuntimeError, "media reached"):
                Node().create(**inputs)
            load.assert_called_once_with("[]")


if __name__ == "__main__":
    unittest.main()

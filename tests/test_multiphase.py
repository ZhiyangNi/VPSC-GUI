from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "repo" / "src"))

from vpsc_gui.app import (  # noqa: E402
    ProjectState,
    collect_vpsc_case_dependencies,
    find_output_file_groups,
    find_phase_output_file,
    parse_vpsc8_in,
    prepare_run_dir,
)


def test_bcc_fcc_two_phase_example_parses_dependencies_and_stages(tmp_path):
    case = ROOT / "examples" / "BCC_FCC_Tension"
    info = parse_vpsc8_in(case / "vpsc8.in")
    assert info.nph == 2
    assert [ph.crystal_file for ph in info.phases[:2]] == ["FCC.sx", "BCC.sx"]
    assert [ph.texture_file for ph in info.phases[:2]] == ["FCC_texture.tex", "BCC_texture.tex"]
    assert info.process_files == ["tension"]

    deps, warnings = collect_vpsc_case_dependencies(case, info)
    dep_names = {p.name for p in deps}
    assert {"FCC.sx", "BCC.sx", "FCC_texture.tex", "BCC_texture.tex", "tension"}.issubset(dep_names)
    assert warnings == []

    state = ProjectState(base_dir=case, vpsc_in=Path("vpsc8.in"), run_root=tmp_path)
    run_dir = prepare_run_dir(state, info)
    for name in dep_names:
        assert (run_dir / name).is_file()
    assert (run_dir / "app_dependency_report.txt").is_file()


def test_phase_resolved_output_discovery_accepts_native_names_and_padding(tmp_path):
    for name in ["ACT_PH2.OUT", "ACT_PH1.OUT", "TEX_PH02.OUT", "TEX_PH1.OUT", "STR_STR.OUT"]:
        (tmp_path / name).write_text("0 0\n", encoding="utf-8")
    groups = find_output_file_groups(tmp_path)
    assert [p.name for p in groups["ACT"]] == ["ACT_PH1.OUT", "ACT_PH2.OUT"]
    assert [p.name for p in groups["TEX"]] == ["TEX_PH1.OUT", "TEX_PH02.OUT"]
    assert find_phase_output_file(tmp_path, "TEX", 2).name == "TEX_PH02.OUT"
    assert groups["STR_STR"][0].name == "STR_STR.OUT"

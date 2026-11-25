"""Contract and deliverable tests for Day 14 student-facing MD, HTML, PDF, and walkthrough images."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _count_prose_words(text: str) -> int:
    """Count learner-facing prose words per vlearn-lab-authoring/references/step-depth-contract.md."""
    cleaned = re.sub(r"```.*?```", " ", text, flags=re.S)
    cleaned = re.sub(
        r":::(checklist|chart|diagram|image|video|repository|code-tabs).*?^:::",
        " ",
        cleaned,
        flags=re.S | re.M,
    )
    cleaned = re.sub(r":::callout[^\n]*\n(.*?)^:::", r"\1", cleaned, flags=re.S | re.M)
    cleaned = "\n".join(ln for ln in cleaned.splitlines() if not ln.strip().startswith("|"))
    cleaned = "\n".join(ln for ln in cleaned.splitlines() if not ln.lstrip().startswith("#"))
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    return len(re.findall(r"\S+", cleaned))


class TestStudentGuideDeliverables(unittest.TestCase):
    STUDENT_MD = REPO_ROOT / "K5-L23-Day14-PointPillars-Prelabel.md"
    STUDENT_HTML = REPO_ROOT / "K5-L23-Day14-PointPillars-Prelabel.html"
    STUDENT_PDF = REPO_ROOT / "K5-L23-Day14-PointPillars-Prelabel.pdf"
    WALKTHROUGH_DIR = REPO_ROOT / "walkthrough_images"

    def _assert_clean_student_text(self, text: str, label: str) -> None:
        stripped_org = re.sub(r"VinUni-AI20k", "", text, flags=re.IGNORECASE)
        for forbidden in (
            "ai20k",
            "coachonly",
            "Labels-VF",
            "15OYxx",
            "1eSdpP",
            "DEMO_E2E_01",
            "QA_STAGE2_SEED",
            "qad14",
            "bonus/",
            "aithucchien",
            "REPORT.md",
        ):
            self.assertNotIn(
                forbidden.lower(),
                stripped_org.lower(),
                f"Forbidden secret/coach token {forbidden!r} found in {label}",
            )
        for pat in (
            r"(?:--seed|D14_SEED\s*=)\s*[\"']?(?!<SEED>|<[^>]+>|\.\.\.)[A-Za-z0-9_-]{3,}",
            r"seed_tag=[0-9a-fA-F]{6,}",
            r"ghp_[A-Za-z0-9]{20,}",
        ):
            self.assertIsNone(re.search(pat, text), f"Secret/SEED pattern {pat!r} matched in {label}")

    def test_student_markdown_contract_widgets_and_images(self) -> None:
        self.assertTrue(self.STUDENT_MD.is_file(), "Missing student Markdown guide")
        text = self.STUDENT_MD.read_text(encoding="utf-8")
        self._assert_clean_student_text(text, self.STUDENT_MD.name)

        yaml_match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
        self.assertIsNotNone(yaml_match, "Missing YAML front matter")
        yaml_block = yaml_match.group(1)
        body = text[yaml_match.end():]
        keys = set(re.findall(r"^([A-Za-z0-9_]+):", yaml_block, re.M))
        allowed = {
            "id", "day", "title", "description", "outcomes",
            "prerequisites", "requiredTools", "commonErrors", "requiresSubmission", "workMode",
        }
        self.assertTrue(keys <= allowed, f"Unsupported YAML keys: {keys - allowed}")
        self.assertIn('day: "D14"', yaml_block)
        self.assertIn('workMode: "individual"', yaml_block)
        self.assertIn("requiresSubmission: true", yaml_block)

        # Zero raw HTML outside fenced code and inline backticks
        no_fences = re.sub(r"```.*?```", "", body, flags=re.S)
        no_inline = re.sub(r"`[^`]+`", "", no_fences)
        raw_html = re.findall(r"</?[A-Za-z][A-Za-z0-9_-]*(?:\s[^>]*)?>", no_inline)
        self.assertEqual(raw_html, [], f"Raw HTML found in student MD: {raw_html}")

        # Only allowed widgets
        directives = re.findall(r"^:::([a-z-]+)(?:\{|$)", no_fences, re.M)
        self.assertTrue(directives, "Expected Codelabs directives")
        self.assertTrue(set(directives) <= {"callout", "checklist", "code-tabs", "repository", "image"})

        # Exact disk parity with walkthrough_images/
        imgs_md = re.findall(r':::image\{src="([^"]+)"', body)
        imgs_disk = sorted(p.relative_to(REPO_ROOT).as_posix() for p in self.WALKTHROUGH_DIR.iterdir() if p.is_file())
        self.assertGreater(len(imgs_disk), 0)
        for src in imgs_md:
            self.assertTrue((REPO_ROOT / src).is_file(), f"Missing image referenced in MD: {src}")
        self.assertEqual(set(imgs_md), set(imgs_disk), "Orphan or unreferenced walkthrough_images in MD")

        # Key required lab facts
        for required in (
            "https://github.com/VinUni-AI20k/K5-L23-Day14-PointPillars-Prelabel",
            "K5-DAY14-HoVaTen-MSSV",
            "ref_B.json",
            "Shape",
            "Track",
            "--org",
            "exit 137",
            "host.docker.internal:8080",
            "key=vals",
        ):
            self.assertIn(required, text)

    def test_student_markdown_step_depth_and_reader_blocks_budgets(self) -> None:
        text = self.STUDENT_MD.read_text(encoding="utf-8")
        yaml_match = re.match(r"^---\r?\n(.*?)\r?\n---\r?\n", text, re.S)
        self.assertIsNotNone(yaml_match)
        body = text[yaml_match.end():]

        sections = re.split(r"(?m)^## ", body)[1:]
        self.assertEqual(len(sections), 5)

        pha0_words = _count_prose_words(sections[0])
        self.assertTrue(180 <= pha0_words <= 300, f"Pha 0 prose {pha0_words} outside 180-300")

        for idx in range(1, 5):
            sec = sections[idx]
            title = sec.splitlines()[0]
            words = _count_prose_words(sec)
            self.assertTrue(500 <= words <= 1000, f"Pha {idx} ({title}) prose {words} outside 500-1000")
            for sub in re.split(r"(?m)^### ", sec)[1:]:
                st = sub.splitlines()[0]
                sw = _count_prose_words(sub)
                if st.startswith("Bản chất"):
                    self.assertTrue(120 <= sw <= 200, f"{st}: {sw} outside 120-200")
                elif st.startswith("Giải nghĩa"):
                    self.assertTrue(100 <= sw <= 160, f"{st}: {sw} outside 100-160")
                elif st.startswith("Lựa chọn thiết kế"):
                    self.assertTrue(100 <= sw <= 180, f"{st}: {sw} outside 100-180")
                elif st.startswith("Checkpoint"):
                    self.assertTrue(80 <= sw <= 160, f"{st}: {sw} outside 80-160")

        no_fences = re.sub(r"```.*?```", "", body, flags=re.S)
        for cm in re.finditer(r":::callout\{([^}]*)\}\n(.*?)^:::", no_fences, re.S | re.M):
            cw = len(cm.group(2).split())
            self.assertTrue(25 <= cw <= 70, f"Callout {cm.group(1)} has {cw} words (expected 25-70)")

        for clm in re.finditer(r":::checklist\{([^}]*)\}\n(.*?)^:::", no_fences, re.S | re.M):
            items = [ln for ln in clm.group(2).splitlines() if ln.strip().startswith("- [")]
            self.assertTrue(3 <= len(items) <= 7, f"Checklist item count {len(items)} outside 3-7")
            for item in items:
                iw = len(item.split()) - 2
                self.assertTrue(5 <= iw <= 16, f"Checklist item {item!r} has {iw} words (expected 5-16)")

        for im in re.finditer(r":::image\{([^}]*)\}", no_fences):
            alt = re.search(r'alt="([^"]*)"', im.group(1)).group(1)
            cap = re.search(r'caption="([^"]*)"', im.group(1)).group(1)
            self.assertTrue(8 <= len(alt.split()) <= 20, f"Alt length out of 8-20w: {alt!r}")
            self.assertTrue(8 <= len(cap.split()) <= 25, f"Caption length out of 8-25w: {cap!r}")

        for slop in ("khám phá", "mạnh mẽ", "toàn diện", "dễ dàng", "seamless", "robust", "best practice"):
            self.assertNotIn(slop.lower(), text.lower(), f"Anti-slop token {slop!r} found in MD")

    def test_student_html_offline_timer_images_and_print_mode(self) -> None:
        self.assertTrue(self.STUDENT_HTML.is_file(), "Missing student HTML guide")
        html = self.STUDENT_HTML.read_text(encoding="utf-8")
        self._assert_clean_student_text(html, self.STUDENT_HTML.name)

        self.assertIsNone(re.search(r'<link[^>]+href=["\']https?://', html, re.I))
        self.assertIsNone(re.search(r'<script[^>]+src=["\']https?://', html, re.I))
        self.assertNotIn("fonts.googleapis.com", html)

        for token in (
            "const TOTAL = 240;",
            'id="bStart"',
            'id="bViewAll"',
            'id="bTheme"',
            'id="bSound"',
            'id="bPresent"',
            'id="bFull"',
            'id="bReset"',
            'id="bSync"',
            'id="refSection"',
            "openLightbox",
            "print-mode",
            "@media print",
            "wrapNoSplit",
            '.replace(/"/g, "&quot;")',
        ):
            self.assertIn(token, html)
        self.assertNotIn('<span class="cm"># text</span>', html)
        archive_html = re.search(r'<div id="deepArchive" hidden>.*?<div class="ref" id="refSection">', html, re.S).group(0)
        self.assertNotIn("**", archive_html, "Unparsed '**' markdown found in HTML deepArchive")
        for cm_inner in re.findall(r'<pre><span class="cm">([^<]*)</span>', archive_html):
            self.assertTrue(cm_inner.endswith("\n"), f"Missing trailing newline in comment span: {cm_inner!r}")

        # Every subToc target must exist in deepArchive and duration sum must equal 240
        phases = json.loads(re.search(r"const PHASES = (\[.*?\]);", html, re.S).group(1))
        self.assertEqual(sum(p["dur"] for p in phases), 240)
        deep_ids = set(re.findall(r'<h4 id="([^"]+)"', html))
        for p in phases:
            for sid, _ in p["subToc"]:
                self.assertIn(sid, deep_ids, f"Missing subToc ID {sid!r} in HTML")

        imgs_html = re.findall(r'<img src="(walkthrough_images/[^"]+)"', html)
        imgs_disk = sorted(p.relative_to(REPO_ROOT).as_posix() for p in self.WALKTHROUGH_DIR.iterdir() if p.is_file())
        for src in imgs_html:
            self.assertTrue((REPO_ROOT / src).is_file(), f"Missing image referenced in HTML: {src}")
        self.assertEqual(set(imgs_html), set(imgs_disk), "Orphan or unreferenced walkthrough_images in HTML")

    def test_student_pdf_exists_and_is_valid_a4(self) -> None:
        self.assertTrue(self.STUDENT_PDF.is_file(), "Missing student PDF guide")
        raw = self.STUDENT_PDF.read_bytes()
        self.assertGreater(len(raw), 100_000)
        self.assertTrue(raw.startswith(b"%PDF-"))
        self.assertIn(b"%%EOF", raw[-1024:])
        try:
            import pymupdf
        except ImportError:
            return
        doc = pymupdf.open(str(self.STUDENT_PDF))
        self.assertGreaterEqual(len(doc), 10)
        self.assertAlmostEqual(doc[0].rect.width, 595.92, delta=2.0)
        self.assertAlmostEqual(doc[0].rect.height, 842.88, delta=2.0)
        total_imgs = sum(len(doc[i].get_images()) for i in range(len(doc)))
        imgs_disk = [p for p in self.WALKTHROUGH_DIR.iterdir() if p.is_file()]
        self.assertEqual(total_imgs, len(imgs_disk))
        full_text = "\n".join(doc[i].get_text("text") for i in range(len(doc)))
        self.assertIn("Trang 1 /", full_text)
        self.assertNotIn("**", full_text, "Literal '**' found in exported PDF text")
        self._assert_clean_student_text(full_text, self.STUDENT_PDF.name)


if __name__ == "__main__":
    unittest.main()

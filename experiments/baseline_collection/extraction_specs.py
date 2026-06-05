# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registry of extraction specs (prompt configurations) for WARC extraction.

Each spec has a short ``spec_id`` that becomes the GCS namespace key:

    gs://{regional_bucket}/documents/baseline_llm_extraction/{spec_id}/data-{warc_hash}/

The ``spec_id`` *is* the contract — no content hashing. If you iterate a prompt
mid-run, bump the id (``v1`` → ``v1b``) so outputs land in a fresh namespace.

Special case: the legacy spec ``low_quality`` writes to the *unprefixed* path
``documents/baseline_llm_extraction/data-{warc_hash}/`` because that's where
the original (pre-registry) extraction landed. Path resolution honors this
via ``LEGACY_SPEC_ID``.

Adding a new spec
-----------------
The simplest path is to call ``make_default_spec(spec_id, spec_text, ...)``:
it reuses the canonical DSPy-style system message and user template, only
swapping in your custom rules text. For full control, construct
``ExtractionSpec`` directly with your own ``system_message`` and
``extraction_template`` (template must contain a ``{example}`` placeholder).
"""

from dataclasses import dataclass

# Canonical DSPy-style system message and user template. Inlined here rather
# than imported from experiments.rephraser.extraction_sft_recipe to avoid
# pulling in the heavy datakit/fray import chain when launching the
# orchestrator locally. Keep in sync with the constants in that file.
DEFAULT_SYSTEM_MESSAGE = (
    "Your input fields are:\n"
    "1. `html` (str): \n"
    "2. `extraction_spec` (str):\n"
    "Your output fields are:\n"
    "1. `text` (str):\n"
    "All interactions will be structured in the following way, "
    "with the appropriate values filled in.\n\n"
    "[[ ## html ## ]]\n{html}\n\n"
    "[[ ## extraction_spec ## ]]\n{extraction_spec}\n\n"
    "[[ ## text ## ]]\n{text}\n\n"
    "[[ ## completed ## ]]\n"
    "In adhering to this structure, your objective is: \n"
    "        Extract the main content text from a given HTML document."
)

DEFAULT_USER_TEMPLATE_FMT = (
    "[[ ## html ## ]]\n{{example}}\n\n"
    "[[ ## extraction_spec ## ]]\n{spec}\n\n"
    "Respond with the corresponding output fields, "
    "starting with the field `[[ ## text ## ]]`, "
    "and then ending with the marker for `[[ ## completed ## ]]`."
)

# Spec id of the original pre-registry extraction. Its data lives at the
# unprefixed GCS path; path resolution special-cases this id to read/write
# there. Other specs nest under their id.
LEGACY_SPEC_ID = "low_quality"


@dataclass(frozen=True)
class ExtractionSpec:
    """One extraction prompt configuration.

    Attributes:
        spec_id: Short URL-safe key. Becomes the GCS subdir under
            ``documents/baseline_llm_extraction/`` (except for
            ``LEGACY_SPEC_ID``, which maps to the unprefixed path).
        system_message: Chat-template system message.
        extraction_template: User-message template; must contain
            ``{example}`` where the HTML is substituted.
        description: Human-readable note about this spec's intent.
    """

    spec_id: str
    system_message: str
    extraction_template: str
    description: str = ""

    def __post_init__(self) -> None:
        if "{example}" not in self.extraction_template:
            raise ValueError(f"extraction_template for spec_id={self.spec_id!r} must contain '{{example}}'")
        if not self.spec_id or "/" in self.spec_id or self.spec_id.startswith("_"):
            raise ValueError(f"spec_id must be non-empty, contain no '/', and not start with '_': {self.spec_id!r}")


def make_default_spec(spec_id: str, spec_text: str, description: str = "") -> ExtractionSpec:
    """Build a spec using the canonical DSPy-style system message + user template.

    Most specs only need to vary the rules text; this helper handles the rest.

    Escaping: curly braces in ``spec_text`` are auto-escaped before
    substitution so authors can copy raw text from source files (e.g.
    ``{word1|word2|word3}`` for spinner template examples) without thinking
    about Python's format-string syntax. The escape survives both the
    construction-time format (which substitutes ``{spec}``) and the runtime
    ``template.format(example=...)`` call in the worker.
    """
    escaped = spec_text.replace("{", "{{").replace("}", "}}")
    return ExtractionSpec(
        spec_id=spec_id,
        system_message=DEFAULT_SYSTEM_MESSAGE,
        extraction_template=DEFAULT_USER_TEMPLATE_FMT.format(spec=escaped),
        description=description,
    )


# ---------------------------------------------------------------------------
# Spec definitions
# ---------------------------------------------------------------------------

# The original extraction prompt — produced the data at the unprefixed legacy
# path. Kept here for completeness so the dashboard and any future re-runs of
# this spec have a single source of truth.
_LOW_QUALITY_RULES = (
    "Extract the content from this HTML page as clean text. Follow all rules below.\n\n"
    "1. Extract the full page content in reading order. Keep all explanatory text, "
    "discussion, and comments that add substantive information.\n"
    "2. Remove boilerplate: navigation bars, footers, sidebars, ads, share buttons, "
    "related links, breadcrumbs, cookie banners, and user interface elements.\n"
    "3. Preserve all technical content exactly as written: code, math notation, "
    "formulas, tables, and data.\n"
    "4. Decode HTML entities to their plain characters (e.g. &amp; to &, &lt; to <, "
    "&gt; to >). Remove any raw HTML tags.\n"
    "5. Every sentence in your output must come from the source page. Do not add, "
    "invent, or embellish content.\n"
    "6. Output exactly [NO_USEFUL_CONTENT] if ANY of these apply:\n"
    "   - Login, signup, paywall, error page, or empty page\n"
    "   - Index page, directory listing, search results, or navigation-only page\n"
    "   - The page has under ~50 words of substantive content after removing boilerplate"
)


# quality_extraction_v8 — stricter quality bar with bullet-by-bullet filtering
# (navigation/aggregation, site machinery, fragments, media wrappers, commerce,
# raw data dumps, too-short, incoherent). Source:
# /Users/michaelryan/Documents/School/Stanford/Research/small-rephraser/prompts/quality_extraction_v8.txt
_MED_QUALITY_RULES = (
    "Extract the content from this HTML page as clean text. Apply all rules below.\n\n"
    "1. **Quality bar.** The page must be real written content — coherent prose that conveys "
    "information, ideas, narrative, or experience. Acceptable: articles, explanations, analyses, "
    "reviews with substantive commentary, tutorials, manuals, reference entries, documentation, "
    "and substantive forum or comment discussions. If the page does not pass, output exactly "
    "[NO_USEFUL_CONTENT].\n\n"
    "   **Walk through each bullet below in order**, checking whether the page matches it. "
    "If any bullet applies, output [NO_USEFUL_CONTENT] and do not extract.\n\n"
    "   - **Navigation or aggregation** — exists to point elsewhere (forum indexes, category "
    "pages, tag clouds, search results, sitemaps, archive-by-date pages, contributor lists, "
    "topic-aggregation pages of one-line summaries). A tag archive showing one truncated post "
    "excerpt still fails — the page must contain at least one full post.\n"
    "   - **Site machinery** — login, signup, paywall, cookie banner, captcha, error page, "
    '"page not found", redirect notice, terms of service, privacy policy, subscription/'
    'notification form ("send to a friend", email signup), or instructions for using the '
    "site itself. Pages whose main content is hidden behind a login wall fail even when the "
    'page shows a "you must be logged in" prompt.\n'
    "   - **Fragments** — Tumblr-style image-reblog streams, social timelines, photo-blog "
    'feeds, lists of one-liner microposts, image gallery captions, personal "about me" '
    'pages, and repost stubs that consist of a brief excerpt plus a "Read on:", "Go to '
    'Source", "View original", or affiliate link to elsewhere. Pages of fragments and '
    "metadata rather than coherent passages.\n"
    "   - **Wrapper around non-textual media** — a video, audio, slideshow, photo, or "
    "interactive widget where the page exists to host the media rather than be read. Includes "
    "image-only social posts (Blingee/Pinterest-style), single-photo pages with a one-line "
    "caption, video-player pages with only a title/byline around the video, and blog posts "
    "whose body is a short reaction (a sentence or two) around an embedded video, image, or "
    "share.\n"
    '   - **Commerce or product page** — pages with prices, availability, "Add to Cart", '
    "or shipping/stock indicators. Vendor-written descriptions, specs, ingredients, and care "
    "instructions are commerce metadata, not content — they do not save the page. Keep only "
    "if the page contains an extended editorial passage about the product (a third-party "
    "review, comparison, or analysis), not vendor sales copy. Also includes **employer or "
    'company profile pages** ("Quick Look", "Company Overview") that are mostly company '
    "name, industry, size, and a bullet list of services or job categories.\n"
    "   - **Raw data dump** — structured data without surrounding narrative. This includes "
    "both rows of data (rosters, statistics tables, file indexes, contact directories, "
    "database exports) **and single-record entries** (museum specimen cards, IP whois / BGP "
    "route lookups, dictionary entries that are only a definition line, badge / user-profile "
    "/ member pages). An article that uses tables to support its prose is fine.\n"
    "   - **Too short** — count the substantive sentences on the page after removing "
    "boilerplate, navigation, ads, comments, links, dates, and reposted excerpts. **If the "
    "remaining substantive prose is fewer than 5 complete sentences, the page fails.** Apply "
    'this strictly: a 1-paragraph news brief, a 2-sentence "X just opened in Y" announcement, '
    'a single-event calendar entry, "this section has moved" notice, "no results found" / '
    '"no products matching" / "click here to proceed" placeholder — all fail, regardless '
    "of how grammatical or informative-looking the few sentences are.\n"
    "   - **Incoherent** — gibberish, garbled encoding, machine-translated nonsense, or "
    "spinner templates throughout.\n\n"
    "First determine if the page clears the quality bar. If it does, follow the following "
    "rules to extract the content.\n\n"
    "2. **Extract the full page content in reading order**. Keep explanatory text, discussion, "
    "and comments that add substantive information. Begin your output directly with the page "
    "content.\n"
    "3. **Remove boilerplate**: navigation, footers, sidebars, ads, share buttons, related "
    "links, breadcrumbs, cookie banners, and UI elements. Do not output framework markers or "
    "metadata tags.\n"
    "4. **Preserve all technical content** exactly as written: code, math, formulas, tables, "
    "and data. Preserve original line breaks and the structure of code blocks.\n"
    "5. Decode HTML entities to plain characters (e.g. &amp; → &, &#8217; → '). Remove raw "
    "HTML tags.\n"
    "6. **Do not add, invent, or embellish content.** Every sentence must come from the source "
    "page.\n"
    "7. For pages with multiple authors or speakers (forums, reviews, comments), preserve who "
    "said what — include usernames or speaker labels.\n"
    "8. If the page contains content spinner templates like {word1|word2|word3}, pick the "
    "first option. If most of the page is spinners, output [NO_USEFUL_CONTENT]."
)


_MED_LOW_QUALITY_RULES = (
    "Extract the main content from the provided HTML into clean Markdown.\n\n"
    "First, check if the page should be rejected. Output exactly [NO_USEFUL_CONTENT] "
    "if ANY of these apply:\n"
    "- Not primarily in English\n"
    "- Login, signup, account, checkout, paywall, or subscribe page\n"
    '- Error page, captcha, cookie wall, bot check, or "session expired"\n'
    "- Empty or near-empty page, directory index, or navigation-only page\n"
    '- User profile, member page, or "who posted" page\n'
    "- Image gallery or photo album listing without articles\n"
    "- Search results page with no actual results\n"
    "- Page where the main content is behind a login wall or paywall\n"
    "- Product listing, gift card, or e-commerce page with prices/availability\n"
    "- Social media post that is just an image or a single short caption\n"
    "- Blog tag page, category page, or archive page that only lists post titles and teasers\n"
    "- After removing boilerplate, the remaining useful text would be under ~100 words\n\n"
    "If the page passes, extract with these rules:\n"
    "- Output Markdown only. No commentary or analysis.\n"
    "- Preserve original wording. Do not summarize or rewrite.\n"
    "- Remove boilerplate: navbars, footers, sidebars, ads, share buttons, related links, "
    "breadcrumbs.\n"
    "- Preserve all technical content exactly: code blocks verbatim with language tags, "
    "math/LaTeX using $$ delimiters, chemical formulas, tables.\n"
    "- Do not truncate or simplify content due to length.\n"
    "- Include comments/replies only if they add real information (answers, corrections).\n"
    "- Start with the page title as a top-level heading if available."
)


_HIGH_QUALITY_RULES = (
    "Extract the content from this HTML page as clean text. Apply all rules below.\n"
    "\n"
    "1. **EXTREME quality bar — keep almost nothing.** This filter is designed to retain only "
    "the very best educational and informative content; only about 1 in 10 pages should pass. "
    "The **default is to output [NO_USEFUL_CONTENT]**.\n"
    "\n"
    "   **HARD GATES — check first, in order. If any gate fails, immediately output "
    "[NO_USEFUL_CONTENT] and stop.**\n"
    "   - **Gate A — Language.** Is the body of the page written primarily in English? If the "
    "main prose is in any other language (French, Spanish, German, Chinese, Russian, etc.), "
    "output [NO_USEFUL_CONTENT]. Quoted phrases, names, citations, or short snippets in another "
    "language are fine; a page whose main prose is non-English is not — even if the topic looks "
    "substantive.\n"
    "   - **Gate B — Prose depth.** Are there roughly 300 or more words of real *prose* — "
    "flowing sentences and paragraphs that explain, argue, or teach? Table cells, formula tokens, "
    "code-only listings, navigation labels, whois fields, stat lines, and metadata do NOT count "
    'toward the word budget. A page that hits 300 "words" only by counting structured-data '
    "fields fails this gate.\n"
    "\n"
    "   A page that clears both gates is kept ONLY if all four of these are also true:\n"
    "\n"
    "   - **Teaches the reader.** The page explains a concept, mechanism, method, or topic; "
    "develops an argument with evidence and examples; reports original analysis; or delivers "
    "substantive technical or instructional material. It must convey real knowledge of the world, "
    "not just headlines, facts, or summaries.\n"
    "   - **Carefully written.** Clear, structured, professional or academic-quality prose. Not "
    "casual, not fragmentary, not marketing copy, not auto-generated, not template-filled.\n"
    "   - **Self-contained value.** A reader who lands here learns meaningfully from this page "
    "alone — not from clicking through to something else.\n"
    "   - **Not matched by any FAILS bullet below.**\n"
    "\n"
    "   **Walk through each FAILS bullet below in order**, checking whether the page matches it. "
    "If any matches, output [NO_USEFUL_CONTENT] and do not extract.\n"
    "\n"
    "   Categories that typically qualify (still subject to all criteria above):\n"
    "   - Long-form journalism with original reporting and analysis — including editorial tech "
    "journalism or product reporting that includes quotes, context, history, or analysis (even "
    "when the subject is a company's product launch)\n"
    "   - Educational or explanatory articles (science, math, history, engineering, medicine, "
    "philosophy, economics, etc.)\n"
    "   - Tutorials, how-tos, and documentation with real teaching — instructions plus the "
    "reasoning behind them\n"
    "   - Encyclopedia and reference entries written with substantive prose\n"
    "   - **Substantive technical forum, mailing-list, or Q&A threads** — e.g., a Stack Overflow "
    "question with a thorough accepted answer, a numpy/scipy mailing-list discussion working "
    "through a real problem, a developer forum thread that reaches a meaningful technical "
    'conclusion. The "casual forum threads" FAILS bullet does NOT apply when the discussion is '
    "technical and develops the topic.\n"
    "   - Research papers, white papers, technical writeups\n"
    "   - Essays and analyses that develop an argument with evidence — including substantive "
    "opinion or commentary on a personal blog or social platform like Tumblr/Medium/Substack\n"
    "   - Code accompanied by genuine teaching, not just a snippet\n"
    "   - Technical blogs sharing real domain expertise\n"
    "   - Institutional or organizational descriptions with substantive prose (academic program "
    "pages with curriculum and outcomes, organizational reports, conference or event summaries "
    "with real narrative)\n"
    "   - Religious or scriptural texts presenting substantive narrative or doctrine (e.g., a "
    "Bible chapter, Quran sura, philosophical canon — numbered-verse format is fine)\n"
    "\n"
    "   Categories that FAIL — filter even if the writing is fine:\n"
    '   - **News briefs, sports recaps, and transaction roundups** — "X happened, said Y" '
    'reporting without analysis, depth, or context. Includes local sports recaps ("West Side '
    'beat Cumberland 14-4"), team transaction logs ("Signed pitcher X to a minor-league '
    'contract"), and short news items even when several are stitched together on one page.\n'
    "   - **Personal lifestyle or diary blogging** — daily-life updates, travelogues, opinion "
    "pieces without intellectual substance. (A personal blog with substantive analysis or "
    "argument is fine — judge by content, not platform.)\n"
    "   - **Light reviews — including aggregator pages of light reviews** — \"I tried it, it's "
    'good" without genuine analysis or comparison. A page that strings together several short '
    "product blurbs is still light reviews, not substantive content.\n"
    "   - **Reference stubs and structured-data dumps** — one-paragraph wiki entries, "
    "definitions, factoid lists, player stat cards, whois/BGP/DNS records, raw lookup results, "
    "single-formula pages. If the page is mostly a table, a record, or one isolated fact, it "
    'fails — fastText-style classifiers may rate this content highly because it "looks '
    'structured," but it is not teaching prose.\n'
    "   - **Casual forum threads** — chit-chat, brief Q&A without resolution, social exchanges "
    "that don't reach a meaningful conclusion. Technical mailing-list or Q&A threads that DO "
    "develop the topic are covered by the qualifying bullet above and are NOT this.\n"
    "   - **Vendor sales copy or purely promotional material** — product descriptions, ad copy, "
    '"buy now" pages. (Institutional descriptions of academic programs, mission statements '
    "with real content, organizational reports, or editorial journalism about a product launch "
    "are NOT this; judge by whether the page sells a product or describes/teaches.)\n"
    "   - **Mostly un-decoded HTML / escape soup** — pages where the visible text is dominated "
    "by raw HTML entities (`&nbsp;`, `&lt;`, `&amp;`, `&#NNN;`), escape sequences, or malformed "
    "markup that survived the page render. Even if the underlying content might be code or text, "
    "a page presented this way is junk for training.\n"
    '   - **"No results" / search-empty pages** — "your search returned no results", "no '
    'items match", "0 records found" — even when they list related links or alternative '
    "searches alongside.\n"
    "   - **Anything that fails a normal quality filter** — commerce pages, navigation or "
    "aggregation surfaces, login walls or other site machinery, fragments and microposts, "
    "wrappers around non-textual media, raw data dumps, empty placeholders, incoherent text\n"
    "\n"
    "   **When in doubt, output [NO_USEFUL_CONTENT].** Reject borderline pages aggressively; "
    "only clear winners pass.\n"
    "\n"
    "First determine if the page clears the EXTREME quality bar. If it does, follow the "
    "following rules to extract the content.\n"
    "\n"
    "2. **Extract the full page content in reading order**. Keep explanatory text, discussion, "
    "and comments that add substantive information. Begin your output directly with the page "
    "content.\n"
    "3. **Remove boilerplate**: navigation, footers, sidebars, ads, share buttons, related "
    "links, breadcrumbs, cookie banners, and UI elements. Do not output framework markers or "
    "metadata tags.\n"
    "4. **Preserve all technical content** exactly as written: code, math, formulas, tables, "
    "and data. Preserve original line breaks and the structure of code blocks.\n"
    "5. Decode HTML entities to plain characters (e.g. &amp; → &, &#8217; → '). Remove raw "
    "HTML tags.\n"
    "6. **Do not add, invent, or embellish content.** Every sentence must come from the source "
    "page.\n"
    "7. For pages with multiple authors or speakers (forums, reviews, comments), preserve who "
    "said what — include usernames or speaker labels.\n"
    "8. If the page contains content spinner templates like {word1|word2|word3}, pick the first "
    "option. If most of the page is spinners, output [NO_USEFUL_CONTENT]."
)


SPECS: dict[str, ExtractionSpec] = {
    "low_quality": make_default_spec(
        spec_id="low_quality",
        spec_text=_LOW_QUALITY_RULES,
        description=(
            "Original extraction prompt; produced the legacy unprefixed data. "
            "This spec_id maps to the unprefixed GCS path; new runs with --spec "
            "low_quality will continue extending that namespace."
        ),
    ),
    "med_low_quality": make_default_spec(
        spec_id="med_low_quality",
        spec_text=_MED_LOW_QUALITY_RULES,
        description=(
            "v6.txt — markdown-output extraction with English-only check, login/paywall/"
            "error/profile/gallery/commerce/tag-archive rejects, <100 word minimum."
        ),
    ),
    "med_quality": make_default_spec(
        spec_id="med_quality",
        spec_text=_MED_QUALITY_RULES,
        description=(
            "quality_extraction_v8: stricter quality bar — explicit bullet-by-bullet "
            "filtering for nav/aggregation, site machinery, fragments, media wrappers, "
            "commerce, raw data dumps, <5-sentence pages, incoherence."
        ),
    ),
    "high_quality": make_default_spec(
        spec_id="high_quality",
        spec_text=_HIGH_QUALITY_RULES,
        description=(
            "extreme_quality_v5: hard gates (English, ≥300 prose words) + four positive "
            "criteria (teaches, careful prose, self-contained, no FAILS) + explicit FAILS "
            "list. Designed to keep ~1 in 10 pages."
        ),
    ),
}


def get_spec(spec_id: str) -> ExtractionSpec:
    """Look up a spec by id. Raises ValueError with the known set on miss."""
    if spec_id not in SPECS:
        raise ValueError(
            f"Unknown spec_id={spec_id!r}. Known: {sorted(SPECS)}. " f"Add it to SPECS in extraction_specs.py."
        )
    return SPECS[spec_id]


import re
import json
from difflib import SequenceMatcher

import streamlit as st
import streamlit.components.v1 as components
import torch

from transformers import T5Tokenizer, T5ForConditionalGeneration
from symspellpy import SymSpell, Verbosity


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Autocorrect Tool",
    page_icon="A",
    layout="wide"
)

MODEL_NAME = "huytd189/grammar-t5-small"

st.session_state.setdefault("input_text", "")
st.session_state.setdefault("output_text", "")


# ==========================================
# LOAD SPELL CHECKER
# ==========================================

@st.cache_resource(show_spinner=False)
def load_spellchecker():

    spell = SymSpell(
        max_dictionary_edit_distance=2,
        prefix_length=7
    )

    import importlib.resources

    dictionary_path = (
        importlib.resources.files("symspellpy")
        / "frequency_dictionary_en_82_765.txt"
    )

    if not spell.load_dictionary(
        str(dictionary_path),
        term_index=0,
        count_index=1
    ):
        raise RuntimeError(
            "English spelling dictionary could not load."
        )

    return spell


# ==========================================
# LOAD AI MODEL
# ==========================================

@st.cache_resource(show_spinner=False)
def load_model():

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    model = T5ForConditionalGeneration.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return tokenizer, model


# ==========================================
# TEXT PATTERN
# ==========================================

WORD_PATTERN = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?"
)


# ==========================================
# COMMON SPELLING FIXES
# ==========================================

COMMON_SPELLING_FIXES = {
    "goin": "going",
    "tommorow": "tomorrow",
    "recieved": "received",
    "mesage": "message",
    "definately": "definitely",
    "becuase": "because",
    "seperate": "separate",
    "wierd": "weird",
    "occured": "occurred",
    "untill": "until",
    "thier": "their",
    "freind": "friend",
    "enviroment": "environment",
    "langauge": "language",
}


# ==========================================
# WORD AND ALPHABET COUNTS
# ==========================================

def count_words(text):
    return len(text.split())


def count_alphabets(text):
    return sum(
        1 for character in text
        if character.isalpha()
    )


# ==========================================
# SPELLING CORRECTION
# ==========================================

def preserve_case(original, corrected):

    if original.isupper():
        return corrected.upper()

    if original.istitle() or original[0].isupper():
        return corrected.capitalize()

    return corrected.lower()


def spell_correct_text(text, spellchecker):

    def replace_word(match):

        original = match.group(0)
        lower_word = original.lower()

        if len(original) <= 2:
            return original

        if original.isupper():
            return original

        previous_text = text[:match.start()]
        previous_line = previous_text.rsplit("\n", 1)[-1]

        sentence_start = (
            not previous_line.strip()
            or bool(
                re.search(
                    r'[.!?]\s*["\'”’)\]]*$',
                    previous_line
                )
            )
        )

        if original[0].isupper() and not sentence_start:
            return original

        if lower_word in COMMON_SPELLING_FIXES:

            return preserve_case(
                original,
                COMMON_SPELLING_FIXES[lower_word]
            )

        suggestions = spellchecker.lookup(
            lower_word,
            Verbosity.CLOSEST,
            max_edit_distance=2,
            include_unknown=True
        )

        if not suggestions:
            return original

        best = suggestions[0]

        if best.distance == 0 or best.distance > 2:
            return original

        return preserve_case(
            original,
            best.term
        )

    return WORD_PATTERN.sub(
        replace_word,
        text
    )


# ==========================================
# SAFE GRAMMAR CHANGES
# ==========================================

ALLOWED_GRAMMAR_CHANGES = {
    ("has", "have"),
    ("has", "had"),
    ("have", "has"),
    ("have", "had"),

    ("tell", "tells"),
    ("tell", "told"),
    ("tells", "told"),

    ("go", "goes"),
    ("go", "went"),
    ("go", "gone"),
    ("goes", "went"),

    ("eat", "eats"),
    ("eat", "ate"),
    ("see", "saw"),
    ("make", "made"),
    ("buy", "bought"),
    ("bring", "brought"),
    ("take", "took"),
    ("think", "thought"),
    ("know", "knew"),
    ("come", "came"),
    ("leave", "left"),
    ("write", "wrote"),
    ("run", "ran"),
    ("say", "said"),
    ("do", "does"),
    ("do", "did"),
    ("does", "did"),
    ("study", "studies"),
    ("study", "studied"),

    ("improved", "improve"),
    ("improved", "improving"),
    ("playing", "play"),
    ("working", "work"),

    ("is", "are"),
    ("are", "is"),
    ("am", "is"),
    ("was", "were"),
    ("were", "was"),
    ("is", "was"),
    ("are", "were"),

    ("dont", "don't"),
    ("doesnt", "doesn't"),
    ("didnt", "didn't"),
}

SAFE_INSERTIONS = {
    "a", "an", "the",
    "is", "am", "are", "was", "were",
    "be", "been", "being",
    "do", "does", "did",
    "have", "has", "had",
    "to", "of", "in", "on", "at",
    "for", "with", "from", "by",
    "and", "but", "or", "so", "because",
    "that", "who", "which", "while", "if",
    "as", "not",
    "my", "your", "our", "their",
    "his", "her", "its",
    "i", "we", "they", "he", "she", "it", "you"
}


# ==========================================
# MEANING PRESERVATION
# ==========================================

def preserve_meaning(original, generated):

    original_matches = list(
        WORD_PATTERN.finditer(original)
    )

    generated_matches = list(
        WORD_PATTERN.finditer(generated)
    )

    original_words = [
        match.group(0).lower()
        for match in original_matches
    ]

    generated_words = [
        match.group(0).lower()
        for match in generated_matches
    ]

    matcher = SequenceMatcher(
        None,
        original_words,
        generated_words,
        autojunk=False
    )

    for operation, i1, i2, j1, j2 in matcher.get_opcodes():

        if operation == "equal":
            continue

        source_words = original_words[i1:i2]
        output_words = generated_words[j1:j2]

        if operation == "insert":

            if not all(
                word in SAFE_INSERTIONS
                for word in output_words
            ):
                return original

            continue

        if operation == "delete":

            if not all(
                word in SAFE_INSERTIONS
                for word in source_words
            ):
                return original

            continue

        if len(source_words) != len(output_words):
            return original

        for source_word, output_word in zip(
            source_words,
            output_words
        ):

            if source_word == output_word:
                continue

            if (
                source_word,
                output_word
            ) not in ALLOWED_GRAMMAR_CHANGES:
                return original

    return generated


# ==========================================
# SPLIT LONG TEXT
# ==========================================

def split_text(text, max_words=80):

    paragraphs = [
        paragraph.strip()
        for paragraph in text.splitlines()
        if paragraph.strip()
    ]

    all_chunks = []

    for paragraph in paragraphs:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            paragraph
        )

        current_chunk = []

        for sentence in sentences:

            words = sentence.split()

            if len(words) > max_words:

                if current_chunk:
                    all_chunks.append(
                        " ".join(current_chunk)
                    )

                    current_chunk = []

                for start in range(
                    0,
                    len(words),
                    max_words
                ):

                    all_chunks.append(
                        " ".join(
                            words[start:start + max_words]
                        )
                    )

                continue

            proposed = " ".join(
                current_chunk + [sentence]
            )

            if (
                len(proposed.split()) > max_words
                and current_chunk
            ):

                all_chunks.append(
                    " ".join(current_chunk)
                )

                current_chunk = [sentence]

            else:
                current_chunk.append(sentence)

        if current_chunk:
            all_chunks.append(
                " ".join(current_chunk)
            )

    return all_chunks


# ==========================================
# CLEAN OUTPUT
# ==========================================

def clean_output(text):

    text = text.strip()

    if len(text) >= 2 and (
        (text.startswith("'") and text.endswith("'"))
        or (text.startswith('"') and text.endswith('"'))
    ):

        text = text[1:-1].strip()

    return text


# ==========================================
# AI GRAMMAR CORRECTION
# ==========================================

def grammar_correct_chunk(chunk, tokenizer, model):

    prompt = (
        "Correct grammar and punctuation only. "
        "Preserve every original content word and meaning. "
        "Do not paraphrase or rewrite. "
        f"Text: {chunk}"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.inference_mode():

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=256,
            num_beams=4,
            early_stopping=True
        )

    generated = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    generated = clean_output(generated)

    if not generated:
        return chunk

    return preserve_meaning(
        chunk,
        generated
    )


# ==========================================
# BASIC CAPITALIZATION
# ==========================================

def fix_capitalization(text):

    text = re.sub(
        r"^(\s*)([a-z])",
        lambda match: (
            match.group(1) + match.group(2).upper()
        ),
        text
    )

    text = re.sub(
        r"\bi\b",
        "I",
        text
    )

    text = re.sub(
        r"\benglish\b",
        "English",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r'([.!?]\s+)([a-z])',
        lambda match: (
            match.group(1) + match.group(2).upper()
        ),
        text
    )

    return text


# ==========================================
# COMPLETE CORRECTION PROCESS
# ==========================================

def correct_text(text, spellchecker, tokenizer, model):

    spelling_fixed = spell_correct_text(
        text,
        spellchecker
    )

    chunks = split_text(spelling_fixed)

    corrected_chunks = []

    for chunk in chunks:

        corrected = grammar_correct_chunk(
            chunk,
            tokenizer,
            model
        )

        corrected_chunks.append(corrected)

    grammar_fixed = "\n\n".join(
        corrected_chunks
    )

    final_text = spell_correct_text(
        grammar_fixed,
        spellchecker
    )

    final_text = fix_capitalization(
        final_text
    )

    return final_text


# ==========================================
# CLEAR RESULT WHEN INPUT CHANGES
# ==========================================

def clear_result():
    st.session_state["output_text"] = ""


# ==========================================
# MIDNIGHT NAVY + ELECTRIC BLUE DESIGN
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

:root {
    --background: #0B1020;
    --panel: #18243B;
    --border: #30415F;
    --text: #F3F6FF;
    --muted: #91A3C5;
    --blue: #557BFF;
    --blue-hover: #4167E8;
}

.stApp {
    background:
        radial-gradient(
            circle at 85% 0%,
            #182746 0%,
            #0B1020 48%
        );
    color: var(--text);
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #111A2B;
    border-right: 1px solid #263752;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: #F3F6FF !important;
}

p, label {
    color: #B3C0D8;
}

[data-testid="stTextArea"] textarea {
    background: #18243B !important;
    color: #F3F6FF !important;
    border: 1px solid #30415F !important;
    border-radius: 13px !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}

[data-testid="stTextArea"] textarea:focus {
    border: 1px solid #557BFF !important;
    box-shadow: 0 0 0 1px #557BFF !important;
}

.stButton > button {
    background: linear-gradient(
        90deg,
        #4167E8,
        #557BFF
    ) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    min-height: 45px;
    font-weight: 700 !important;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #3657CA,
        #4167E8
    ) !important;
}

.metric-line {
    color: #91A3C5;
    font-size: 13px;
    padding-top: 5px;
}

hr {
    border-color: #263752 !important;
}

[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown("# Autocorrect Tool")

st.caption(
    "Write or paste your text and let AI improve it."
)

st.write("")


# ==========================================
# TWO TEXT BOXES
# ==========================================

left, right = st.columns(
    2,
    gap="large"
)


# ==========================================
# LEFT BOX - ORIGINAL TEXT
# ==========================================

with left:

    st.markdown("### Your Text")

    original_text = st.text_area(
        "Original text",
        key="input_text",
        height=310,
        placeholder=(
            "Start typing or paste your paragraph here..."
        ),
        label_visibility="collapsed",
        on_change=clear_result
    )

    st.markdown(
        f"""
        <div class="metric-line">
            Words: <b>{count_words(original_text)}</b>
            &nbsp; | &nbsp;
            Alphabets: <b>{count_alphabets(original_text)}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    correct_clicked = st.button(
        "Correct Text",
        type="primary",
        use_container_width=True
    )


# ==========================================
# CORRECTION ACTION
# ==========================================

if correct_clicked:

    if not original_text.strip():

        st.warning("Please enter some text first.")

    else:

        try:

            with st.spinner("Loading spelling dictionary..."):

                spellchecker = load_spellchecker()

            with st.spinner("Loading AI grammar model..."):

                tokenizer, model = load_model()

            with st.spinner("Checking spelling and grammar..."):

                corrected = correct_text(
                    original_text,
                    spellchecker,
                    tokenizer,
                    model
                )

            st.session_state["output_text"] = corrected

            st.rerun()

        except Exception as error:

            st.error(
                f"Correction failed: {error}"
            )


# ==========================================
# RIGHT BOX - CORRECTED TEXT
# ==========================================

with right:

    heading_col, copy_col = st.columns([5, 1])

    with heading_col:

        st.markdown("### Corrected Text")

    with copy_col:

        copy_value = st.session_state.get(
            "output_text",
            ""
        )

        safe_text = (
            json.dumps(copy_value, ensure_ascii=False)
            .replace("<", "\\u003c")
            .replace(">", "\\u003e")
            .replace("&", "\\u0026")
        )

        copy_html = """
        <html>
        <body style="
            margin:0;
            padding:0;
            background:transparent;
        ">

        <button id="copyButton"
            __DISABLED__
            style="
                background:#557BFF;
                color:white;
                border:0;
                border-radius:9px;
                padding:9px 15px;
                font-size:13px;
                font-weight:700;
                cursor:pointer;
                width:100%;
                height:38px;
                opacity:__OPACITY__;
            ">
            Copy
        </button>

        <script>

        const textToCopy = __TEXT__;

        const button = document.getElementById("copyButton");

        button.addEventListener("click", async function() {

            if (!textToCopy) return;

            try {

                await navigator.clipboard.writeText(textToCopy);

                button.innerText = "Copied!";

            } catch(error) {

                const temporary = document.createElement("textarea");

                temporary.value = textToCopy;

                document.body.appendChild(temporary);

                temporary.select();

                const success = document.execCommand("copy");

                temporary.remove();

                button.innerText = success
                    ? "Copied!"
                    : "Copy failed";

            }

            setTimeout(function() {

                button.innerText = "Copy";

            }, 1500);

        });

        </script>
        </body>
        </html>
        """

        copy_html = (
            copy_html
            .replace("__TEXT__", safe_text)
            .replace(
                "__DISABLED__",
                "disabled" if not copy_value else ""
            )
            .replace(
                "__OPACITY__",
                "0.45" if not copy_value else "1"
            )
        )

        components.html(
            copy_html,
            height=43,
            scrolling=False
        )

    corrected_text = st.text_area(
        "Corrected result",
        key="output_text",
        height=310,
        placeholder=(
            "Your corrected text will appear here. "
            "You can also edit it..."
        ),
        label_visibility="collapsed"
    )

    st.markdown(
        f"""
        <div class="metric-line">
            Words: <b>{count_words(corrected_text)}</b>
            &nbsp; | &nbsp;
            Alphabets: <b>{count_alphabets(corrected_text)}</b>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# FOOTER
# ==========================================

st.write("")
st.divider()

st.caption(
    "Autocorrect Tool · AI-powered English text correction"
)

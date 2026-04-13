"""
MD&A Sentiment Dashboard
========================
Run: streamlit run webapp/app.py
Requires: webapp/final_df.parquet  (built by webapp/final_data.ipynb)
"""

import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

# ── Global Plotly theme: white bg, no gridlines, keep default color palette ──
_clean_tpl = go.layout.Template()
_clean_tpl.layout = go.Layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=go.layout.XAxis(showgrid=False, linecolor="#e5e7eb", linewidth=1),
    yaxis=go.layout.YAxis(showgrid=False, linecolor="#e5e7eb", linewidth=1),
    font=dict(family="Inter, system-ui, sans-serif"),
)
pio.templates["dash_clean"] = _clean_tpl
pio.templates.default = "plotly+dash_clean"

WEBAPP_DIR = Path(__file__).parent

COMPANY_PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#f03b20",
    "#a6cee3",
    "#b2df8a",
    "#fb9a99",
    "#fdbf6f",
    "#cab2d6",
    "#ffff99",
    "#b15928",
    "#6a3d9a",
    "#33a02c",
    "#e31a1c",
]


def dn(c: str) -> str:
    """Display name: replace underscores with spaces."""
    return c.replace("_", " ")


def fmt_score(v):
    return "-" if pd.isna(v) else f"{v:+.2f}"


st.set_page_config(
    page_title="MD&A Sentiment",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Wrap every Plotly chart in a card */
    div[data-testid="stPlotlyChart"] {
        background: #ffffff !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 14px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05) !important;
        overflow: hidden !important;
        padding: 8px 4px 2px 4px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading data…")
def load_data() -> pd.DataFrame:
    final_path = WEBAPP_DIR / "final_df.parquet"
    if not final_path.exists():
        st.error(
            "final_df.parquet not found. Run webapp/final_data.ipynb first to build it."
        )
        st.stop()

    df = pd.read_parquet(final_path)
    df = df.rename(
        columns={
            "company": "company_name",
            "score": "sentiment_score",
            "label": "finbert_label",
            "sentence": "text",
            "pos": "pos_prob",
            "neg": "neg_prob",
            "neu": "neu_prob",
            "topic_weight": "topic_prob",
        }
    )
    df["topic_label"] = df["topic_label"].fillna("Uncategorised")
    df["year"] = df["year"].astype(int)
    return df


merged = load_data()

# ── Derived universe ────────────────────────────────────────────────────────
all_cos = sorted(merged["company_name"].dropna().unique().tolist())
avail_yrs = sorted(merged["year"].dropna().unique().tolist())
all_topics = sorted(merged["topic_label"].dropna().unique().tolist())

co_colors = {
    c: COMPANY_PALETTE[i % len(COMPANY_PALETTE)] for i, c in enumerate(all_cos)
}

# ── Sector / industry classification ─────────────────────────────────────────
# Maps company_name (as it appears in the dataset) to a sector string.
# Companies not in this map default to "Technology".
_SECTOR_OF: dict = {
    **dict.fromkeys([
        "NVIDIA", "AMD", "Intel", "Broadcom", "QUALCOMM", "Texas_Instruments",
        "Microchip_Technology", "Micron_Technology", "Marvell_Technology",
        "ON_Semiconductor", "NXP_Semiconductors", "Skyworks_Solutions", "Qorvo",
        "Lattice_Semiconductor", "Silicon_Labs", "Monolithic_Power_Systems",
        "Analog_Devices", "Cirrus_Logic", "MACOM", "Ambarella",
        "Allegro_MicroSystems", "Alpha__amp__Omega_Semiconductor", "CEVA",
        "Navitas_Semiconductor", "Semtech", "Power_Integrations", "SiTime",
        "MaxLinear", "Pixelworks", "Everspin_Technologies", "GCT_Semiconductor",
        "Intchains_Group", "Magnachip_Semiconductor", "Vishay_Intertechnology",
        "Wolfspeed", "indie_Semiconductor", "Silvaco_Group", "Aeluma",
        "Lightwave_Logic", "Coherent_Corp_", "Quicklogic", "nLIGHT", "Synaptics",
        "Arm_Holdings", "Astera_Labs", "Credo_Technology", "Impinj", "Arteris",
        "Allient", "Lumentum", "TE_Connectivity", "Qnity_Electronics",
        "Mobix_Labs", "Geospace_Technologies", "Sensata_Technologies",
    ], "Semiconductors"),
    **dict.fromkeys([
        "Applied_Materials", "Lam_Research", "KLA", "Axcelis_Technologies",
        "Cohu", "Entegris", "FormFactor", "Ichor_Systems", "Photronics",
        "SkyWater_Technology", "Kulicke_and_Soffa_Industries", "Veeco",
        "UCT__Ultra_Clean_Holdings_", "Amtech_Systems", "Atomera",
        "ACM_Research", "Aehr_Test_Systems", "Onto_Innovation", "PDF_Solutions",
        "Amkor_Technology", "Keysight",
    ], "Semiconductor Equipment"),
    **dict.fromkeys([
        "Microsoft", "Salesforce", "Workday", "ServiceNow", "Adobe", "Intuit",
        "Oracle", "HubSpot", "Atlassian", "Asana", "DocuSign", "Box__Inc_",
        "MongoDB", "Elastic_NV", "GitLab", "JFrog", "PagerDuty", "Nutanix",
        "Guidewire_Software", "Bentley_Systems", "PTC", "Trimble",
        "Roper_Technologies", "Autodesk", "Cadence_Design_Systems", "Synopsys",
        "FactSet", "Fair_Isaac__FICO_", "Manhattan_Associates",
        "SS_C_Technologies", "Tyler_Technologies", "Paycom", "Paylocity",
        "Automatic_Data_Processing", "Jack_Henry__amp__Associates",
        "Progress_Software", "BlackLine", "Veeva_Systems", "IQVIA",
        "Certara", "Simulations_Plus", "Consensus_Cloud_Solutions", "OneStream",
        "Intapp", "EverCommerce", "Appian", "Pegasystems",
        "CCC_Intelligent_Solutions", "CSG_International",
        "Donnelley_Financial_Solutions", "ePlus", "i3_Verticals", "PAR_Technology",
        "ReposiTrak", "Workiva", "Agilysys", "Samsara", "ServiceTitan",
        "Dayforce", "Blackbaud", "Commvault", "Domo", "Sprinklr", "Braze",
        "Amplitude", "Clearwater_Analytics", "Clarivate", "Rimini_Street",
        "Rackspace_Technology", "AvePoint", "N-Able", "Asure_Software",
        "Alight", "Grid_Dynamics", "EPAM_Systems", "Genpact", "Teradata",
        "Unisys", "IBM", "Xerox", "Conduent", "Diebold_Nixdorf",
        "NCR_Atleos_Corporation", "NCR_Voyix_Corporation", "Leidos",
        "SPS_Commerce", "Confluent", "Cimpress", "Cognyte_Software",
        "LegalZoom", "Thryv", "Xperi", "Mitek_Systems", "ACI_Worldwide",
        "Digimarc", "Cerence", "Dropbox", "Sprout_Social", "Digi_International",
        "Procore", "CoStar_Group", "Freshworks", "Figma", "Klaviyo",
        "Navan", "Toast", "UiPath", "Datadog", "Dynatrace", "Snowflake",
        "Zeta_Global", "ZoomInfo", "Yext", "Semrush", "TechTarget", "ON24",
        "Kaltura", "Chegg", "Coursera", "Udemy", "eGain", "Forian",
        "Research_Solutions", "QXO__Inc_", "Quantum", "Penguin_Solutions",
        "Vertex",
    ], "Software & SaaS"),
    **dict.fromkeys([
        "CrowdStrike", "Palo_Alto_Networks", "Okta", "SentinelOne", "Fortinet",
        "Zscaler", "Rapid7", "Qualys", "Tenable", "Varonis_Systems",
        "NETSCOUT", "BlackBerry", "Gen_Digital", "Identiv", "SailPoint",
        "Telos", "SoundThinking____ShotSpotter_", "Rubrik", "OneSpan",
        "Intellicheck", "Castellum",
    ], "Cybersecurity"),
    **dict.fromkeys([
        "Amazon", "Alphabet__Google_", "Cloudflare", "Fastly", "Akamai",
        "Equinix", "DigitalOcean", "CoreWeave", "Backblaze",
        "Pure_Storage", "NetApp", "Seagate_Technology", "Sandisk",
        "F5", "Arista_Networks", "Cisco", "Calix", "A10_Networks",
        "Harmonic_Inc_", "Sanmina", "Supermicro", "Dell",
        "HP", "Hewlett_Packard_Enterprise", "NETGEAR", "Lantronix",
        "One_Stop_Systems",
    ], "Cloud & Infrastructure"),
    **dict.fromkeys([
        "Meta_Platforms__Facebook_", "Snap", "Reddit", "Nextdoor",
        "Bumble", "Match_Group", "Grindr", "Life360", "Rumble",
        "CuriosityStream", "Scienjoy_Holding_Corporation", "Sohu_com",
        "Yelp", "ZipRecruiter", "QuinStreet", "Taboola_com", "LiveRamp",
        "Digital_Turbine", "Viant_Technology", "DoubleVerify", "Criteo",
        "PubMatic", "AppLovin", "The_Trade_Desk", "Tingo_Group", "Agora_io",
        "MNTN__Inc_", "Teads",
    ], "Social Media & Ad Tech"),
    **dict.fromkeys([
        "PayPal", "Block", "SoFi", "LendingClub", "LendingTree",
        "Marqeta", "Upstart", "Dave_Inc_", "NerdWallet", "Remitly",
        "Payoneer", "Paysign", "Flywire", "Repay_Holdings",
        "Q2", "nCino", "Alkami_Technology", "Forge_Global",
        "Pagaya_Technologies", "Priority_Technology_Holdings",
        "Fiserv", "Global_Payments", "Fidelity_National_Information_Services",
        "Futu_Holdings", "UP_Fintech__Tiger_Brokers_", "WM_Technology",
        "Blend_Labs", "Expensify", "Robinhood", "MediaAlpha",
        "Circle_Internet_Group", "Coinbase", "Ibotta", "Cantaloupe",
        "Claritev", "PROG_Holdings", "Waystar", "Chime_Financial",
        "Lesaka_Technologies", "Blackboxstocks",
    ], "Fintech & Payments"),
    **dict.fromkeys([
        "Strategy____MicroStrategy_", "Bit_Digital", "MARA_Holdings",
        "Riot_Platforms", "CleanSpark", "Core_Scientific", "Cipher_Mining",
        "HIVE_Blockchain_Technologies", "Hut_8", "IREN__Iris_Energy_",
        "TeraWulf", "Applied_Digital", "Bakkt_Holdings",
        "TON_Strategy_Co_", "ALT5_Sigma", "Bitmine_Immersion_Technologies",
        "Chaince_Digital_Holdings",
    ], "Crypto & Blockchain"),
    **dict.fromkeys([
        "eBay", "Etsy", "Wayfair", "MercadoLibre", "Coupang",
        "Chewy", "CarParts_com", "Revolve", "Stitch_Fix", "ThredUp",
        "The_RealReal", "GigaCloud_Technology", "Liquidity_Services",
        "Trip_com", "Booking_Holdings__Booking_com_", "Expedia_Group",
        "Airbnb", "DoorDash", "Instacart__Maplebear_Inc__", "Groupon",
        "1-800-PetMeds", "Sea_Limited", "The_Original_BARK_Company",
    ], "E-commerce & Marketplace"),
    **dict.fromkeys([
        "Electronic_Arts", "Take-Two_Interactive", "Playtika", "Playstudios",
        "Skillz", "Unity_Software", "Corsair_Gaming", "Netflix", "Spotify",
        "Roku", "Shutterstock", "Getty_Images",
    ], "Gaming & Entertainment"),
    **dict.fromkeys([
        "Apple", "Logitech", "Garmin", "Zebra_Technologies", "Protolabs",
        "Latch", "SmartRent", "3D_Systems", "Vuzix", "Arlo_Technologies",
    ], "Hardware & Devices"),
    **dict.fromkeys([
        "Palantir", "C3_AI", "BigBear_ai", "SoundHound_AI", "Tempus_AI",
        "Veritone", "iLearningEngines", "Palladyne_AI", "IonQ",
        "D-Wave_Quantum", "Rigetti_Computing", "Quantum_Computing",
        "Symbotic", "Richtech_Robotics", "Ekso_Bionics", "Velo3D",
        "MicroVision", "Odysight_ai", "Arrive_AI", "Fermi_Inc_",
    ], "AI & Robotics"),
    **dict.fromkeys([
        "Tesla", "Uber", "Lyft", "Mobileye", "Aurora_Innovation",
        "AEye", "Arbe_Robotics", "Ouster", "WeRide", "Red_Cat_Holdings",
        "AgEagle_Aerial_Systems", "Ondas_Holdings", "Rekor_Systems",
        "Marti_Technologies", "PowerFleet", "Airship_AI",
        "Duos_Technologies_Group",
    ], "EV & Mobility"),
    **dict.fromkeys([
        "8x8", "RingCentral", "Zoom", "Five9", "Twilio", "Tucows",
        "Lumen_Technologies", "Comtech_Telecommunications", "Spire_Global",
        "NextNav", "Synchronoss", "Spok_Holdings", "Genasys", "KORE",
        "AST_SpaceMobile", "InterDigital", "Weave_Communications",
    ], "Communications"),
    **dict.fromkeys([
        "Teladoc_Health", "Doximity", "American_Well", "Phreesia",
        "Omnicell", "Veradigm", "Talkspace", "LifeStance_Health_Group",
        "LifeMD", "eHealth", "OptimizeRx", "Schrödinger", "Align_Technology",
        "Hims__amp__Hers_Health", "ANI_Pharmaceuticals", "Absci",
    ], "Healthcare Technology"),
    **dict.fromkeys([
        "Enphase_Energy", "Stem__Inc",
    ], "Clean Energy Technology"),
    **dict.fromkeys([
        "Compass", "reAlpha_Tech", "Zillow",
    ], "Real Estate Technology"),
    **dict.fromkeys([
        "Hippo", "Lemonade", "Porch_Group",
    ], "Insurtech"),
    **dict.fromkeys([
        "CLEAR_Secure", "CS_Disco", "Nerdy__Inc_", "BlackSky_Technology",
        "Gloo_Holdings", "Where_Food_Comes_From", "Red_Violet",
        "WhiteFiber", "Sabre",
    ], "Other Technology"),
}


def companies_in_same_sector(selected_cos: list) -> list:
    """Return all companies in the same sector(s) as the selected companies."""
    sectors = {_SECTOR_OF.get(c, "Technology") for c in selected_cos}
    return sorted(c for c in all_cos if _SECTOR_OF.get(c, "Technology") in sectors)


# ── Latest quarter in dataset ────────────────────────────────────────────────
_q_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
_latest_row = (
    merged[["year", "quarter"]]
    .drop_duplicates()
    .assign(_pk=lambda d: d["year"] * 10 + d["quarter"].map(_q_order))
    .sort_values("_pk")
    .iloc[-1]
)
LATEST_YEAR = int(_latest_row["year"])
LATEST_Q = str(_latest_row["quarter"])


# ── Helper functions ────────────────────────────────────────────────────────
def _filter(df, company=None, year=None, quarter=None, topic=None):
    if company is not None:
        df = df[df["company_name"] == company]
    if year is not None:
        df = df[df["year"] == year]
    if quarter is not None:
        df = df[df["quarter"] == quarter]
    if topic is not None:
        df = df[df["topic_label"] == topic]
    return df


def co_score(company, year, quarter, col="pos_prob"):
    sub = _filter(merged, company=company, year=year, quarter=quarter)
    return float(sub[col].mean()) if len(sub) else np.nan


def co_score_label_filtered(company, year, quarter, col="pos_prob", label="positive"):
    """Average col over sentences where finbert_label == label."""
    sub = _filter(merged, company=company, year=year, quarter=quarter)
    sub = sub[sub["finbert_label"] == label]
    return float(sub[col].mean()) if len(sub) else np.nan


@st.cache_data(show_spinner=False)
def _topic_scores_all_cos_latest():
    """Vectorized: avg pos_prob (pos-sentiment sentences) and avg neg_prob (neg-sentiment sentences)
    per (company, topic) for LATEST_YEAR/LATEST_Q. Used by tab-1 cards. Cached once per session."""
    sub = merged[(merged["year"] == LATEST_YEAR) & (merged["quarter"] == LATEST_Q)]
    pos_scores = (
        sub[sub["sentiment_score"] > 0]
        .groupby(["company_name", "topic_label"])["pos_prob"]
        .mean()
    )
    neg_scores = (
        sub[sub["sentiment_score"] < 0]
        .groupby(["company_name", "topic_label"])["neg_prob"]
        .mean()
    )
    return pos_scores, neg_scores


def portfolio_score(weights, year, quarter, col="pos_prob"):
    ws = wt = 0.0
    for c, w in weights.items():
        s = co_score(c, year, quarter, col=col)
        if not np.isnan(s):
            ws += w * s
            wt += w
    return ws / wt if wt else np.nan


def portfolio_score_label_filtered(
    weights, year, quarter, col="pos_prob", label="positive"
):
    """Weighted avg of col, restricted to sentences with finbert_label == label."""
    ws = wt = 0.0
    for c, w in weights.items():
        s = co_score_label_filtered(c, year, quarter, col=col, label=label)
        if not np.isnan(s):
            ws += w * s
            wt += w
    return ws / wt if wt else np.nan


def topic_scores(company, year, quarter, col="pos_prob"):
    sub = _filter(merged, company=company, year=year, quarter=quarter)
    return sub.groupby("topic_label")[col].mean()


def topic_scores_filtered(company, year, quarter, metric="pos"):
    """
    For 'pos': average pos_prob over sentences where sentiment_score > 0 per topic.
    For 'neg': average neg_prob over sentences where sentiment_score < 0 per topic.
    Denominator = count of qualifying sentences per topic.
    """
    sub = _filter(merged, company=company, year=year, quarter=quarter)
    if metric == "pos":
        sub = sub[sub["sentiment_score"] > 0]
        return sub.groupby("topic_label")["pos_prob"].mean()
    else:
        sub = sub[sub["sentiment_score"] < 0]
        return sub.groupby("topic_label")["neg_prob"].mean()


def portfolio_topic_scores(weights, year, quarter, col="pos_prob"):
    rows = []
    for c, w in weights.items():
        for t, s in topic_scores(c, year, quarter, col=col).items():
            rows.append({"topic": t, "score": s, "w": w})
    if not rows:
        return pd.Series(dtype=float)
    tmp = pd.DataFrame(rows)
    return tmp.groupby("topic").apply(lambda g: np.average(g["score"], weights=g["w"]))


def portfolio_topic_scores_filtered(weights, year, quarter, metric="pos"):
    rows = []
    for c, w in weights.items():
        for t, s in topic_scores_filtered(c, year, quarter, metric=metric).items():
            rows.append({"topic": t, "score": s, "w": w})
    if not rows:
        return pd.Series(dtype=float)
    tmp = pd.DataFrame(rows)
    return tmp.groupby("topic").apply(lambda g: np.average(g["score"], weights=g["w"]))


_SENTIMENT_LABELS = {"positive", "negative"}


def co_score_nonzero(company, year, quarter):
    """Average sentiment_score over sentiment-dense sentences (positive/negative labels only)."""
    sub = _filter(merged, company=company, year=year, quarter=quarter)
    sub = sub[sub["finbert_label"].isin(_SENTIMENT_LABELS)]
    return float(sub["sentiment_score"].mean()) if len(sub) else np.nan


def portfolio_score_nonzero(weights, year, quarter):
    """Weighted average of co_score_nonzero across companies."""
    ws = wt = 0.0
    for c, w in weights.items():
        s = co_score_nonzero(c, year, quarter)
        if not np.isnan(s):
            ws += w * s
            wt += w
    return ws / wt if wt else np.nan


def ranked_company_scores(companies, year, quarter, topic=None, col="pos_prob"):
    rows = []
    for c in companies:
        sub = _filter(merged, company=c, year=year, quarter=quarter, topic=topic)
        if len(sub):
            rows.append(
                {"company": c, "display": dn(c), "score": float(sub[col].mean())}
            )
    if not rows:
        return pd.DataFrame(columns=["company", "display", "score"])
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def ranked_company_scores_dense(companies, year, quarter, topic=None):
    """Sentiment-dense ranking (pos/neg labels only). Always includes ALL companies;
    companies with no data score 0.0 so they still appear in the chart."""
    rows = []
    for c in companies:
        sub = _filter(merged, company=c, year=year, quarter=quarter, topic=topic)
        sub = sub[sub["finbert_label"].isin(_SENTIMENT_LABELS)]
        score = float(sub["sentiment_score"].mean()) if len(sub) else 0.0
        rows.append({"company": c, "display": dn(c), "score": score})
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def ranked_company_pos_dense(companies, year, quarter, topic=None):
    """Rank by avg (pos_prob - neg_prob) over sentiment-dense (pos+neg labelled) sentences.
    Always includes all companies; 0.0 if no data."""
    rows = []
    for c in companies:
        sub = _filter(merged, company=c, year=year, quarter=quarter, topic=topic)
        sub = sub[sub["finbert_label"].isin(_SENTIMENT_LABELS)]
        score = float((sub["pos_prob"] - sub["neg_prob"]).mean()) if len(sub) else 0.0
        rows.append({"company": c, "display": dn(c), "score": score})
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def apply_top_bottom(df, mode, n, score_col="score"):
    if not len(df):
        return df
    if mode == "Top":
        return df.nlargest(n, score_col)
    if mode == "Bottom":
        return df.nsmallest(n, score_col)
    return df


def topic_keyword_table(df, topics, top_n_words=10):
    rows = []
    for topic in topics:
        texts = df[df["topic_label"] == topic]["text"].dropna().astype(str)
        texts = texts[texts.str.strip().str.len() > 0]
        keywords = ""
        if len(texts):
            try:
                vec = CountVectorizer(stop_words="english", max_features=5000)
                x = vec.fit_transform(texts)
                if x.shape[1] > 0:
                    freqs = np.asarray(x.sum(axis=0)).ravel()
                    terms = np.array(vec.get_feature_names_out())
                    # Remove numeric placeholder tokens (e.g. "num", "num0", "num123")
                    _keep = np.array([not re.match(r"^num\d*$", t) for t in terms])
                    freqs, terms = freqs[_keep], terms[_keep]
                    top_idx = np.argsort(freqs)[::-1][:top_n_words]
                    keywords = ", ".join(terms[top_idx])
            except ValueError:
                pass
        rows.append({"Topic": topic, "Top words": keywords})
    return pd.DataFrame(rows)


# Added due to issue when there is only 1 company selected
def safe_n_slider(container, label, upper, default, key):
    if upper <= 1:
        container.caption(f"{label}: 1")
        return 1
    return container.slider(label, 1, upper, min(default, upper), key=key)


# Added to assist in finding common topics between companies in peer comparison
def common_topics_for(companies):
    if not companies:
        return []

    topic_sets = []
    for c in companies:
        c_topics = set(
            merged.loc[merged["company_name"] == c, "topic_label"]
            .dropna()
            .unique()
            .tolist()
        )
        topic_sets.append(c_topics)

    return sorted(set.intersection(*topic_sets)) if topic_sets else []


# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("MD&A Sentiment")
    st.success(
        f"Real data loaded · {len(merged):,} sentences · {len(all_cos)} companies"
    )

    _default_cos = ["NVIDIA", "Apple", "Microsoft", "Tesla", "Alphabet__Google_"]
    _default_selection = [c for c in _default_cos if c in all_cos] or all_cos[:5]
    selected = st.multiselect(
        "Companies",
        all_cos,
        default=_default_selection,
        format_func=dn,
    )
    if not selected:
        st.warning("Select at least one company.")
        st.stop()

    st.subheader("Weights")
    n_cos = len(selected)
    default_w = 100 // n_cos

    # Initialise session-state weights; reset stale keys for deselected companies
    active_keys = {f"w_{c}" for c in selected}
    # for k in list(st.session_state.keys()):
    #     if k.startswith("w_") and k not in active_keys:
    #         del st.session_state[k]
    for c in selected:
        if f"w_{c}" not in st.session_state:
            st.session_state[f"w_{c}"] = default_w

    def _sync_weights(changed_c):
        key = f"w_{changed_c}"
        if key not in st.session_state:
            return
        
        new_val = max(0, min(100, st.session_state[key]))
        new_val = max(0, min(100, st.session_state[f"w_{changed_c}"]))
        others = [c for c in selected if c != changed_c]
        if not others:
            return
        old_other_sum = sum(st.session_state.get(f"w_{c}", 0) for c in others)
        remaining = 100 - new_val
        if remaining < 0:
            remaining = 0
            st.session_state[f"w_{changed_c}"] = 100
        if old_other_sum == 0:
            per = remaining // len(others)
            for c in others:
                st.session_state[f"w_{c}"] = per
        else:
            for i, c in enumerate(others):
                if i < len(others) - 1:
                    st.session_state[f"w_{c}"] = round(
                        st.session_state[f"w_{c}"] / old_other_sum * remaining
                    )
                else:
                    # last one absorbs rounding remainder
                    st.session_state[f"w_{c}"] = remaining - sum(
                        st.session_state[f"w_{o}"] for o in others[:-1]
                    )

    for c in selected:
        st.number_input(
            dn(c),
            min_value=0,
            max_value=100,
            step=1,
            key=f"w_{c}",
            on_change=_sync_weights,
            args=(c,),
        )

    # w_raw = {c: st.session_state[f"w_{c}"] for c in selected}
    w_raw = {c: st.session_state.get(f"w_{c}", default_w) for c in selected}
    total_raw = sum(w_raw.values()) or 1
    weights = {c: v / total_raw for c, v in w_raw.items()}

    pie_fig = go.Figure(
        go.Pie(
            labels=[dn(c) for c in selected],
            values=[weights[c] * 100 for c in selected],
            marker_colors=[co_colors[c] for c in selected],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f}%<extra></extra>",
            hole=0.35,
        )
    )
    pie_fig.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        height=240,
        showlegend=False,
    )
    st.plotly_chart(pie_fig, use_container_width=True)

    score_col = "sentiment_score"
    score_label = "Avg sentiment score (pos − neg)"

    st.subheader("Period")
    period_mode = st.selectbox("Scope", ["All-time", "Specific quarter"], index=0)
    if period_mode == "Specific quarter":
        sel_year = st.selectbox("Year", list(reversed(avail_yrs)))
        sel_q = st.selectbox("Quarter", ["Q4", "Q3", "Q2", "Q1"])
        period_label = f"{sel_year} {sel_q}"
    else:
        sel_year = None
        sel_q = None
        period_label = "All-time"


# ── Header ───────────────────────────────────────────────────────────────────
st.title("MD&A Sentiment Dashboard")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Corpus EDA",
        "Topic Trends",
        "Peer Comparison",
        "Single Filing Drill-Down",
        "Topic Modeling",
    ]
)


# ── Tab 1: Corpus EDA ────────────────────────────────────────────────────────
with tab1:
    st.subheader("Portfolio Overview")
    card_period_label = f"{LATEST_YEAR} {LATEST_Q}"

    _is_pos = True
    _t_metric = "pos"
    t_col_label = "avg pos (sentences w/ pos sentiment)"
    # ── Compute radar / table helpers (still needed for those sections) ──────────
    _all_pos_ts, _all_neg_ts = _topic_scores_all_cos_latest()

    _port_cos = set(weights.keys())
    _p_pos_by_topic = (
        _all_pos_ts[_all_pos_ts.index.get_level_values("company_name").isin(_port_cos)]
        .groupby("topic_label")
        .mean()
    )
    _p_neg_by_topic = (
        _all_neg_ts[_all_neg_ts.index.get_level_values("company_name").isin(_port_cos)]
        .groupby("topic_label")
        .mean()
    )

    # ── Card values: avg(pos_prob − neg_prob) over dense sentences, latest quarter ──
    _latest_dense = merged[
        (merged["year"] == LATEST_YEAR)
        & (merged["quarter"] == LATEST_Q)
        & (merged["finbert_label"].isin(_SENTIMENT_LABELS))
    ]
    # Per-company avg(pos-neg), then mean across companies (equal weight)
    _port_co_posneg = (
        _latest_dense[_latest_dense["company_name"].isin(_port_cos)]
        .groupby("company_name")
        .apply(lambda g: float((g["pos_prob"] - g["neg_prob"]).mean()))
    )
    p_diff = float(_port_co_posneg.mean()) if len(_port_co_posneg) else np.nan

    _all_co_posneg = _latest_dense.groupby("company_name").apply(
        lambda g: float((g["pos_prob"] - g["neg_prob"]).mean())
    )
    all_stocks_diff = float(_all_co_posneg.mean()) if len(_all_co_posneg) else np.nan

    def _card(label, value, color, period=""):
        val_str = f"{value:+.3f}" if not pd.isna(value) else "–"
        sub = (
            f"<div style='font-size:0.72rem;color:#9ca3af;margin-top:2px'>{period}</div>"
            if period
            else ""
        )
        return (
            f"<div style='flex:1;border:1px solid #e5e7eb;border-radius:12px;padding:18px 20px;"
            f"background:#fff'>"
            f"<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.09em;color:#6b7280;"
            f"margin-bottom:8px;text-transform:uppercase'>{label}</div>"
            f"<div style='font-size:2rem;font-weight:800;color:{color};line-height:1'>{val_str}</div>"
            f"{sub}"
            f"</div>"
        )

    _port_card_color = "#16a34a" if (not pd.isna(p_diff) and p_diff >= 0) else "#dc2626"
    _all_card_color = (
        "#16a34a"
        if (not pd.isna(all_stocks_diff) and all_stocks_diff >= 0)
        else "#dc2626"
    )

    _guide_card = (
        "<div style='flex:1;border:1px solid #e5e7eb;border-radius:12px;padding:18px 20px;"
        "background:#fff'>"
        "<div style='font-size:0.65rem;font-weight:700;letter-spacing:0.09em;color:#6b7280;"
        "margin-bottom:10px;text-transform:uppercase'>How to read Avg (Pos − Neg)</div>"
        "<div style='font-size:0.78rem;color:#374151;line-height:1.65'>"
        "<span style='color:#16a34a;font-weight:700'>&#9679; Strong positive &nbsp;&gt; +0.20</span><br>"
        "MD&amp;A language is clearly optimistic — growth, confidence, strong outlook.<br><br>"
        "<span style='color:#6b7280;font-weight:700'>&#9679; Neutral &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;−0.05 to +0.20</span><br>"
        "Balanced or cautious tone — typical for most filings.<br><br>"
        "<span style='color:#dc2626;font-weight:700'>&#9679; Negative &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&lt; −0.05</span><br>"
        "Risk language dominates — losses, headwinds, uncertainty."
        "</div>"
        "</div>"
    )

    cards_html = (
        "<div style='display:flex;gap:14px;margin-bottom:16px'>"
        + _card(
            "Avg (Pos − Neg) · Portfolio",
            p_diff,
            _port_card_color,
            f"{card_period_label} · {len(_port_cos)} companies · equal weight",
        )
        + _card(
            "Avg (Pos − Neg) · All Stocks",
            all_stocks_diff,
            _all_card_color,
            f"{card_period_label} · {len(all_cos)} companies",
        )
        + _guide_card
        + "</div>"
    )
    st.markdown(cards_html, unsafe_allow_html=True)

    # ── Radar + Portfolio Sentiment side-by-side ──────────────────────────────
    _radar_col, _sent_col = st.columns(2, gap="medium")

    # Left: Portfolio Topic Radar
    with _radar_col:
        st.markdown(
            "<span style='font-size:0.68rem;font-weight:700;letter-spacing:0.09em;"
            "color:#6b7280'>❄️ &nbsp;PORTFOLIO TOPIC RADAR · AVG (POS − NEG)</span>",
            unsafe_allow_html=True,
        )
        # Build per-topic (pos−neg) mapped to [0,1] — portfolio simple avg vs all-cos avg
        _radar_topics = sorted(
            _p_pos_by_topic.index.union(_p_neg_by_topic.index).tolist()
        )
        _radar_vals = []
        for _t in _radar_topics:
            _pv = float(_p_pos_by_topic.get(_t, np.nan))
            _nv = float(_p_neg_by_topic.get(_t, np.nan))
            _diff = (_pv - _nv + 1) / 2 if not (np.isnan(_pv) or np.isnan(_nv)) else 0.5
            _radar_vals.append(max(0.0, min(1.0, _diff)))

        if _radar_topics:
            _radar_labels = [dn(t) for t in _radar_topics]
            # close the polygon
            _rv_closed = _radar_vals + [_radar_vals[0]]
            _rl_closed = _radar_labels + [_radar_labels[0]]

            # All-companies avg trace (grey)
            # Reuse _all_pos_ts / _all_neg_ts which cover every company at LATEST_YEAR/LATEST_Q
            _all_pos_by_topic = _all_pos_ts.groupby("topic_label").mean()
            _all_neg_by_topic = _all_neg_ts.groupby("topic_label").mean()
            _all_radar_vals = []
            for _t in _radar_topics:
                _ap = float(_all_pos_by_topic.get(_t, np.nan))
                _an = float(_all_neg_by_topic.get(_t, np.nan))
                _ad = (
                    (_ap - _an + 1) / 2 if not (np.isnan(_ap) or np.isnan(_an)) else 0.5
                )
                _all_radar_vals.append(max(0.0, min(1.0, _ad)))
            _arv_closed = _all_radar_vals + [_all_radar_vals[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=_arv_closed,
                    theta=_rl_closed,
                    fill="toself",
                    fillcolor="rgba(156,163,175,0.12)",
                    line=dict(color="#9ca3af", width=1.5, dash="dot"),
                    name=f"All cos avg (n={len(all_cos)})",
                    hovertemplate="%{theta}<br>All cos avg: %{r:.2f}<extra></extra>",
                )
            )
            fig_radar.add_trace(
                go.Scatterpolar(
                    r=_rv_closed,
                    theta=_rl_closed,
                    fill="toself",
                    fillcolor="rgba(59,130,246,0.18)",
                    line=dict(color="#3b82f6", width=2),
                    name="Portfolio",
                    hovertemplate="%{theta}<br>Portfolio (pos−neg): %{r:.2f}<extra></extra>",
                )
            )
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="white",
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        showticklabels=False,
                        gridcolor="#e5e7eb",
                        linecolor="#e5e7eb",
                    ),
                    angularaxis=dict(
                        gridcolor="#e5e7eb",
                        linecolor="#e5e7eb",
                    ),
                ),
                paper_bgcolor="white",
                plot_bgcolor="white",
                margin=dict(t=20, b=20, l=40, r=40),
                height=360,
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.12,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11),
                ),
            )
            st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("No topic data for radar.")

    # Right: avg(pos−neg) per company — equal weight, consistent with card
    with _sent_col:
        st.markdown(
            f"<span style='font-size:0.68rem;font-weight:700;letter-spacing:0.09em;"
            f"color:#6b7280'>📊 &nbsp;AVG (POS − NEG) PER COMPANY · {card_period_label.upper()}</span>",
            unsafe_allow_html=True,
        )

        # Per-company avg(pos−neg) over dense sentences — reuse _port_co_posneg for latest quarter
        _co_scores = {
            _c: float(_port_co_posneg.loc[_c])
            if _c in _port_co_posneg.index
            else np.nan
            for _c in selected
        }

        # YoY: same metric, previous year same quarter
        def _co_posneg_prev(company, year, quarter):
            sub = _filter(merged, company=company, year=year, quarter=quarter)
            sub = sub[sub["finbert_label"].isin(_SENTIMENT_LABELS)]
            return (
                float((sub["pos_prob"] - sub["neg_prob"]).mean())
                if len(sub)
                else np.nan
            )

        _co_scores_prev = {
            _c: _co_posneg_prev(_c, LATEST_YEAR - 1, LATEST_Q) for _c in selected
        }
        _sorted_cos = sorted(
            selected,
            key=lambda _c: _co_scores[_c] if not np.isnan(_co_scores[_c]) else -999,
            reverse=True,
        )

        # Bar chart: per-company avg(pos−neg), sorted top → bottom
        _bar_labels, _bar_vals, _bar_colors, _bar_text = [], [], [], []
        for _c in reversed(_sorted_cos):
            _cs = _co_scores[_c]
            _cs_prev = _co_scores_prev[_c]
            _yoy_c = (
                _cs - _cs_prev if not (np.isnan(_cs) or np.isnan(_cs_prev)) else np.nan
            )
            _yoy_part = (
                f"  ↑{abs(_yoy_c):.2f}"
                if (not np.isnan(_yoy_c) and _yoy_c >= 0)
                else (f"  ↓{abs(_yoy_c):.2f}" if not np.isnan(_yoy_c) else "")
            )
            _bar_labels.append(dn(_c))
            _bar_vals.append(_cs if not np.isnan(_cs) else 0.0)
            _bar_text.append(f"{_cs:+.3f}{_yoy_part}" if not np.isnan(_cs) else "—")
            _bar_colors.append(
                "#3fb950" if (not np.isnan(_cs) and _cs >= 0) else "#f87171"
            )

        fig_sent_bar = go.Figure(
            go.Bar(
                x=_bar_vals,
                y=_bar_labels,
                orientation="h",
                marker=dict(color=_bar_colors, cornerradius=4, line=dict(width=0)),
                text=_bar_text,
                textposition="outside",
                hovertemplate="%{y}<br>Avg (pos−neg): %{x:+.3f}<extra></extra>",
            )
        )
        _bx_min = min(min(_bar_vals), 0) - 0.05
        _bx_max = max(max(_bar_vals), 0) + 0.18
        fig_sent_bar.update_layout(
            height=max(200, len(_sorted_cos) * 46 + 60),
            xaxis=dict(
                range=[_bx_min, _bx_max],
                zeroline=True,
                zerolinecolor="#e5e7eb",
                showgrid=False,
            ),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=6, b=10, l=10, r=80),
        )
        st.plotly_chart(fig_sent_bar, use_container_width=True)

    # ── Full-width time-series chart ──────────────────────────────────────────
    ctrl1, ctrl2 = st.columns([1.1, 0.9])
    t1_line_mode = ctrl1.selectbox(
        "Line chart filter", ["All", "Top", "Bottom"], key="t1_line_mode"
    )
    max_line_n = max(1, min(10, len(selected)))
    t1_line_n = safe_n_slider(ctrl2, "N companies", max_line_n, 5, "t1_line_n")

    ranked_all = ranked_company_scores(selected, sel_year, sel_q, col=score_col)
    ranked_filtered = apply_top_bottom(ranked_all, t1_line_mode, t1_line_n)
    companies_to_plot = (
        ranked_filtered["company"].tolist() if len(ranked_filtered) else selected
    )

    periods = (
        merged[merged["company_name"].isin(companies_to_plot)]
        .groupby(["year", "quarter", "company_name"])[[score_col]]
        .mean()
        .reset_index()
    )
    if len(periods):
        periods["period_key"] = periods["year"] * 10 + periods["quarter"].map(
            {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        )
        fig = go.Figure()
        for c in companies_to_plot:
            sub = periods[periods["company_name"] == c].sort_values("period_key")
            if not len(sub):
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["year"].astype(str) + " " + sub["quarter"],
                    y=sub[score_col],
                    name=dn(c),
                    mode="lines+markers",
                    line=dict(color=co_colors.get(c, "#1f77b4"), width=2),
                    marker=dict(size=5),
                )
            )
        fig.update_layout(height=420, yaxis_title=score_label, xaxis_title="Period")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No timeline data available.")

    # ── Topic Sentiment Score Per Company ─────────────────────────────────────
    st.divider()

    st.markdown(
        "<span style='font-size:0.75rem;font-weight:600;letter-spacing:0.08em;color:#555'>"
        "TOPIC SENTIMENT SCORE PER COMPANY — ALL TIME"
        "</span>",
        unsafe_allow_html=True,
    )
    _tbl_metric = st.radio(
        "Metric",
        ["Positive (pos)", "Negative (neg)"],
        index=0,
        horizontal=True,
        key="tbl_metric_choice",
        label_visibility="collapsed",
    )
    _is_pos = _tbl_metric == "Positive (pos)"
    _t_metric = "pos" if _is_pos else "neg"
    t_col_label = (
        "avg pos (sentences w/ pos sentiment)"
        if _is_pos
        else "avg neg (sentences w/ neg sentiment)"
    )

    # Filtered computation: pos averages only over positive-sentiment sentences, neg over negative
    p_ts_all = portfolio_topic_scores_filtered(
        weights, sel_year, sel_q, metric=_t_metric
    )
    co_ts = {
        c: topic_scores_filtered(c, sel_year, sel_q, metric=_t_metric) for c in selected
    }

    # Union of topics that have at least one value
    topic_union_all = sorted(
        set(p_ts_all.index.tolist()) | {t for s in co_ts.values() for t in s.index}
    )

    max_topics = len(topic_union_all) or 1
    top_n_topics = st.number_input(
        "Top N topics",
        min_value=1,
        max_value=max_topics,
        value=min(10, max_topics),
        step=1,
        key="top_n_topics",
    )

    # Sort by portfolio avg descending, take top N
    topic_scores_for_sort = {t: p_ts_all.get(t, np.nan) for t in topic_union_all}
    topic_union = sorted(
        topic_union_all,
        key=lambda t: (
            not pd.isna(topic_scores_for_sort[t]),
            topic_scores_for_sort.get(t, -999),
        ),
        reverse=True,
    )[: int(top_n_topics)]

    if topic_union:
        dot_colors = [co_colors[c] for c in selected]
        pct_labels = [f"{int(round(weights[c] * 100))}%" for c in selected]

        def _score_cell(v, bold=False):
            """Render a table cell. Positive metric → green +X.XX; Negative → red -X.XX.
            Only show values that exist (NaN → dash). Hide zeros for positive."""
            if pd.isna(v):
                return "<td style='text-align:center;color:#aaa'>—</td>"
            if _is_pos:
                if v <= 0:
                    return "<td style='text-align:center;color:#aaa'>—</td>"
                color = "#16a34a"
                fmt = f"+{v:.2f}"
            else:
                color = "#dc2626"
                fmt = f"-{v:.2f}"
            fw = "font-weight:700;" if bold else ""
            return (
                f"<td style='text-align:center;color:{color};{fw}'>"
                f"{'<b>' if bold else ''}{fmt}{'</b>' if bold else ''}</td>"
            )

        def _best_worst_cells(topic):
            scores = {c: co_ts[c].get(topic, np.nan) for c in selected}
            valid = {c: s for c, s in scores.items() if not pd.isna(s) and s > 0}
            if not valid:
                return (
                    "<td style='text-align:center;color:#aaa'>—</td>"
                    "<td style='text-align:center;color:#aaa'>—</td>"
                )
            if _is_pos:
                best = max(valid, key=lambda c: valid[c])
                worst = min(valid, key=lambda c: valid[c])
            else:
                # Best = least negative (lowest score), Worst = most negative (highest score)
                best = min(valid, key=lambda c: valid[c])
                worst = max(valid, key=lambda c: valid[c])
            return (
                f"<td style='text-align:center;font-weight:600;color:#16a34a'>{dn(best)}</td>"
                f"<td style='text-align:center;font-weight:600;color:#dc2626'>{dn(worst)}</td>"
            )

        header_cells = "".join(
            f"<th style='text-align:center;color:{dot_colors[i]};white-space:nowrap'>"
            f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            f"background:{dot_colors[i]};margin-right:4px'></span>"
            f"{dn(c)}<br><span style='font-size:0.75rem'>({pct_labels[i]})</span></th>"
            for i, c in enumerate(selected)
        )

        rows_html = ""
        for topic in topic_union:
            wtd = p_ts_all.get(topic, np.nan)
            co_cells = "".join(
                _score_cell(co_ts[c].get(topic, np.nan)) for c in selected
            )
            best_worst = _best_worst_cells(topic)
            rows_html += (
                f"<tr>"
                f"<td style='padding:8px 12px;font-weight:500'>{dn(topic)}</td>"
                f"{co_cells}"
                f"{_score_cell(wtd, bold=True)}"
                f"{best_worst}"
                f"</tr>"
            )

        avg_col_color = "#16a34a" if _is_pos else "#dc2626"
        table_html = f"""
        <style>
        .pwt-table {{border-collapse:collapse;width:100%;font-size:0.875rem}}
        .pwt-table th,.pwt-table td {{padding:7px 10px;border-bottom:1px solid #e5e7eb}}
        .pwt-table th {{font-size:0.75rem;font-weight:600;letter-spacing:0.06em;
                        color:#6b7280;text-transform:uppercase;border-bottom:2px solid #d1d5db;
                        background:#f3f4f6}}
        .pwt-table tr:hover td {{background:#f9fafb}}
        </style>
        <table class='pwt-table'>
          <thead><tr>
            <th style='text-align:left'>Topic</th>
            {header_cells}
            <th style='text-align:center;color:{avg_col_color}'>Portfolio Avg<br>
              <span style='font-weight:400;font-size:0.7rem;color:#6b7280'>({t_col_label})</span></th>
            <th style='text-align:center;color:#16a34a'>Best Stock</th>
            <th style='text-align:center;color:#dc2626'>Worst Stock</th>
          </tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)

        # Overall best/worst asset (aggregated across all topics in the table)
        co_overall = {}
        for c in selected:
            vals = [co_ts[c].get(t, np.nan) for t in topic_union]
            vals = [v for v in vals if not pd.isna(v) and v > 0]
            co_overall[c] = float(np.mean(vals)) if vals else np.nan

        valid_overall = {c: s for c, s in co_overall.items() if not pd.isna(s)}
        if valid_overall:
            if _is_pos:
                overall_best = max(valid_overall, key=lambda c: valid_overall[c])
                overall_worst = min(valid_overall, key=lambda c: valid_overall[c])
                best_fmt = f"+{valid_overall[overall_best]:.2f}"
                worst_fmt = f"+{valid_overall[overall_worst]:.2f}"
            else:
                overall_best = min(valid_overall, key=lambda c: valid_overall[c])
                overall_worst = max(valid_overall, key=lambda c: valid_overall[c])
                best_fmt = f"-{valid_overall[overall_best]:.2f}"
                worst_fmt = f"-{valid_overall[overall_worst]:.2f}"
            n_t = len(topic_union)
            st.markdown(
                f"<div style='margin-top:14px;display:flex;gap:14px'>"
                f"<div style='flex:1;padding:12px 16px;border:1px solid #d1fae5;"
                f"border-radius:8px;background:#f0fdf4'>"
                f"<div style='font-size:0.68rem;font-weight:700;letter-spacing:0.09em;"
                f"color:#065f46;margin-bottom:4px'>OVERALL BEST ASSET</div>"
                f"<div style='font-size:1.25rem;font-weight:800;color:#16a34a'>{dn(overall_best)}</div>"
                f"<div style='font-size:0.78rem;color:#374151;margin-top:2px'>"
                f"Avg: <b style='color:#16a34a'>{best_fmt}</b> across {n_t} topic{'s' if n_t != 1 else ''}</div>"
                f"</div>"
                f"<div style='flex:1;padding:12px 16px;border:1px solid #fee2e2;"
                f"border-radius:8px;background:#fef2f2'>"
                f"<div style='font-size:0.68rem;font-weight:700;letter-spacing:0.09em;"
                f"color:#991b1b;margin-bottom:4px'>OVERALL WORST ASSET</div>"
                f"<div style='font-size:1.25rem;font-weight:800;color:#dc2626'>{dn(overall_worst)}</div>"
                f"<div style='font-size:0.78rem;color:#374151;margin-top:2px'>"
                f"Avg: <b style='color:#dc2626'>{worst_fmt}</b> across {n_t} topic{'s' if n_t != 1 else ''}</div>"
                f"</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    else:
        st.info("No topic scores available for this period.")


# ── Tab 2: Topic Trends ──────────────────────────────────────────────────────
with tab2:
    st.subheader("Topic Trends")
    _t2_default_co = "Microsoft"
    _t2_co_idx = selected.index(_t2_default_co) if _t2_default_co in selected else 0
    t2_co = st.selectbox(
        "Company", selected, index=_t2_co_idx, format_func=dn, key="t2_co"
    )
    co_data = merged[merged["company_name"] == t2_co]

    # Sort topics by number of distinct (year, quarter) periods — richest data first
    _all_co_topics = co_data["topic_label"].dropna().unique().tolist()
    company_topics = sorted(
        _all_co_topics,
        key=lambda t: (
            -int(
                co_data[co_data["topic_label"] == t][["year", "quarter"]]
                .drop_duplicates()
                .shape[0]
            )
        ),
    )
    if not company_topics:
        st.info("No topics available for this company.")
    else:
        _t2_default_topic = "Cash Flow & Operating Expenses"
        if (
            st.session_state.get("t2_topic") not in company_topics
            or st.session_state.get("_t2_co") != t2_co
        ):
            st.session_state["t2_topic"] = (
                _t2_default_topic
                if _t2_default_topic in company_topics
                else company_topics[0]
            )
            st.session_state["_t2_co"] = t2_co
        t2_topic = st.selectbox(
            "Topic",
            company_topics,
            index=company_topics.index(st.session_state["t2_topic"]),
            format_func=dn,
            key="t2_topic",
        )

    t2_trend_view = st.selectbox(
        "Trend chart filter",
        ["Both", "Topic only", "Overall only"],
        key="t2_trend_view",
    )

    def _period_df(df):
        if not len(df):
            return df
        df = df.copy()
        df["period_key"] = df["year"] * 10 + df["quarter"].map(
            {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
        )
        df["period"] = df["year"].astype(str) + " " + df["quarter"]
        return df.sort_values("period_key")

    # Use sentiment-dense sentences only (positive/negative label) for richer signal
    _co_dense = co_data[co_data["finbert_label"].isin(_SENTIMENT_LABELS)]
    trend = _period_df(
        _co_dense[_co_dense["topic_label"] == t2_topic]
        .groupby(["year", "quarter"])[[score_col]]
        .mean()
        .reset_index()
    )
    overall = _period_df(
        _co_dense.groupby(["year", "quarter"])[[score_col]].mean().reset_index()
    )

    # Compute trend status and trough for insight note
    if len(trend) >= 2:
        _delta = float(trend[score_col].iloc[-1] - trend[score_col].iloc[0])
        _min_idx = trend[score_col].idxmin()
        _trough_period = trend.loc[_min_idx, "period"]
        _first_p = trend["period"].iloc[0]
        _last_p = trend["period"].iloc[-1]
        if _delta > 0.08:
            _status_text, _status_bg, _status_fg = "↑ Recovering", "#dcfce7", "#16a34a"
            _insight = (
                f"✅ {dn(t2_co)}'s {dn(t2_topic)} is trending upward"
                f" (trough: {_trough_period}). Cross-check Peer Compare to confirm sector-wide."
            )
        elif _delta < -0.08:
            _status_text, _status_bg, _status_fg = "↓ Declining", "#fee2e2", "#dc2626"
            _insight = (
                f"⚠️ {dn(t2_co)}'s {dn(t2_topic)} is in a downtrend since {_trough_period}."
                f" Monitor for continued deterioration."
            )
        else:
            _status_text, _status_bg, _status_fg = "→ Stable", "#f3f4f6", "#6b7280"
            _insight = (
                f"📊 {dn(t2_co)}'s {dn(t2_topic)} sentiment has been stable"
                f" across {len(trend)} quarters ({_first_p} → {_last_p})."
            )
    else:
        _status_text = _status_bg = _status_fg = None
        _trough_period = "—"
        _insight = ""

    # Chart header
    _badge_html = (
        f"<span style='padding:4px 11px;border-radius:20px;font-size:0.74rem;"
        f"font-weight:600;background:{_status_bg};color:{_status_fg}'>{_status_text}</span>"
        if _status_text
        else ""
    )
    st.markdown(
        f"<div style='display:flex;align-items:flex-start;justify-content:space-between;"
        f"padding:10px 4px 2px'>"
        f"<div>"
        f"<span style='font-size:0.7rem;font-weight:700;letter-spacing:0.09em;color:#6b7280'>"
        f"📈 AVG FINBERT SCORE &nbsp;·&nbsp; {dn(t2_topic).upper()} &nbsp;·&nbsp; {dn(t2_co).upper()}"
        f"</span>"
        f"<div style='font-size:0.73rem;color:#9ca3af;margin-top:2px'>"
        f"Mean FinBERT score of all sentences assigned to this topic cluster, per quarter"
        f"</div></div>"
        f"{_badge_html}</div>",
        unsafe_allow_html=True,
    )

    fig_t = go.Figure()
    if len(overall) and t2_trend_view in ["Both", "Overall only"]:
        fig_t.add_trace(
            go.Scatter(
                x=overall["period"],
                y=overall[score_col],
                name="Overall avg",
                mode="lines",
                line=dict(color="#9ca3af", width=1.6, dash="dot"),
            )
        )
    if len(trend) and t2_trend_view in ["Both", "Topic only"]:
        fig_t.add_trace(
            go.Scatter(
                x=trend["period"],
                y=trend[score_col],
                name=dn(t2_topic),
                mode="lines+markers",
                line=dict(color="#16a34a", width=2.5),
                marker=dict(size=7, color="#16a34a"),
                fill="tozeroy",
                fillcolor="rgba(22,163,74,0.10)",
            )
        )
    fig_t.add_hline(y=0, line_dash="dot", line_color="#d1d5db", line_width=1)
    fig_t.update_layout(
        height=520,
        yaxis_title=score_label,
        xaxis_title="Period",
        yaxis=dict(range=[-1, 1]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig_t, use_container_width=True)

    if _insight:
        st.markdown(
            f"<div style='padding:10px 14px;border:1px solid #e5e7eb;border-radius:6px;"
            f"background:#f9fafb;font-size:0.82rem;color:#374151;margin-top:-8px'>{_insight}</div>",
            unsafe_allow_html=True,
        )

    # ── % of sentences per year (company-level, not topic-specific) ─────────────
    st.markdown(
        f"<span style='font-size:0.7rem;font-weight:700;letter-spacing:0.09em;color:#6b7280'>"
        f"📊 &nbsp;{dn(t2_co).upper()} · OVERALL SENTIMENT DISTRIBUTION BY YEAR"
        f"<span style='font-weight:400;font-size:0.7rem;color:#9ca3af'>"
        f" &nbsp;·&nbsp; all topics combined, not filtered by selected topic</span></span>",
        unsafe_allow_html=True,
    )
    dist = co_data.groupby(["year", "finbert_label"]).size().reset_index(name="n")
    if len(dist):
        dist["total"] = dist.groupby("year")["n"].transform("sum")
        dist["pct"] = dist["n"] / dist["total"] * 100

        label_colors = {
            "positive": "#3fb950",
            "neutral": "#9ca3af",
            "negative": "#f87171",
        }
        fig_bar = go.Figure()
        for lbl in ["positive", "neutral", "negative"]:
            sub = dist[dist["finbert_label"] == lbl]
            fig_bar.add_trace(
                go.Bar(
                    x=sub["year"],
                    y=sub["pct"],
                    name=lbl.capitalize(),
                    marker=dict(
                        color=label_colors[lbl], cornerradius=4, line=dict(width=0)
                    ),
                    hovertemplate="%{x}<br>"
                    + lbl.capitalize()
                    + ": %{y:.2f}%<extra></extra>",
                )
            )
        fig_bar.update_layout(
            barmode="group",
            height=420,
            yaxis_title="% of sentences",
            xaxis_title="Year",
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(t=16, b=40),
        )
        st.plotly_chart(fig_bar, use_container_width=True)


# ── Tab 3: Peer Comparison ───────────────────────────────────────────────────
with tab3:
    st.subheader("Peer Comparison")

    if len(selected) < 2:
        st.info("Select at least 2 companies for peer comparison.")
    else:
        peer_topics = common_topics_for(selected)

        if not peer_topics:
            st.info("No common topics across the selected companies.")
        else:
            _default_peer_topic = "Cash Flow & Operating Expenses"
            # Always reset to the default topic when companies change or on first load
            if (
                st.session_state.get("peer_topic") not in peer_topics
                or st.session_state.get("_peer_companies") != selected
            ):
                st.session_state["peer_topic"] = (
                    _default_peer_topic
                    if _default_peer_topic in peer_topics
                    else peer_topics[0]
                )
                st.session_state["_peer_companies"] = selected
            peer_topic = st.selectbox(
                "Peer topic",
                peer_topics,
                index=peer_topics.index(st.session_state["peer_topic"]),
                format_func=dn,
                key="peer_topic",
            )

            st.markdown(
                f"<div style='padding:14px 18px;background:#f9fafb;border:1px solid #e5e7eb;"
                f"border-radius:12px;margin-bottom:12px'>"
                f"<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.09em;"
                f"color:#6b7280'>📈 SENTIMENT TREND &nbsp;·&nbsp; {dn(peer_topic).upper()}</div>"
                f"<div style='font-size:0.78rem;color:#9ca3af;margin-top:4px'>"
                f"Lines converging = sector-wide signal. One line diverging = idiosyncratic company risk."
                f"</div></div>",
                unsafe_allow_html=True,
            )

            _all_sectors = sorted({_SECTOR_OF.get(c, "Technology") for c in all_cos})
            _DEFAULT_PEER_SECTOR = "Cloud & Infrastructure"
            if "peer_sectors" not in st.session_state:
                st.session_state["peer_sectors"] = (
                    [_DEFAULT_PEER_SECTOR]
                    if _DEFAULT_PEER_SECTOR in _all_sectors
                    else _all_sectors[:1]
                )
            peer_sectors = st.multiselect(
                "Industry",
                _all_sectors,
                key="peer_sectors",
            )
            if not peer_sectors:
                peer_sectors = _all_sectors
            same_sector_cos = sorted(
                c for c in all_cos if _SECTOR_OF.get(c, "Technology") in set(peer_sectors)
            )

            pctrl1, pctrl2 = st.columns([1.1, 0.9])
            if "peer_line_mode" not in st.session_state:
                st.session_state["peer_line_mode"] = "Top"
            peer_line_mode = pctrl1.selectbox(
                "Line chart filter", ["All", "Top", "Bottom"], key="peer_line_mode"
            )
            max_peer_n = max(1, min(10, len(same_sector_cos)))
            peer_line_n = safe_n_slider(
                pctrl2, "N companies (peer line)", max_peer_n, 5, "peer_line_n"
            )

            peer_rank = ranked_company_scores(
                same_sector_cos, sel_year, sel_q, topic=peer_topic, col=score_col
            )
            peer_line_df = apply_top_bottom(peer_rank, peer_line_mode, peer_line_n)
            peer_companies = (
                peer_line_df["company"].tolist() if len(peer_line_df) else same_sector_cos
            )

            if len(peer_rank) == 0:
                st.info("No peer comparison available for this topic.")
            else:
                # fig_peer = go.Figure()
                # for c in peer_companies:
                #     sub = (
                #         merged[
                #             (merged["company_name"] == c) &
                #             (merged["topic_label"] == peer_topic)
                #         ]
                #         .groupby(["year", "quarter"])["sentiment_score"]
                #         .mean()
                #         .reset_index()
                #     )
                #     if not len(sub):
                #         continue
                #     sub["period_key"] = sub["year"] * 10 + sub["quarter"].map(
                #         {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
                #     )
                #     sub["period"] = sub["year"].astype(str) + " " + sub["quarter"]
                #     sub = sub.sort_values("period_key")
                #     fig_peer.add_trace(go.Scatter(
                #         x=sub["period"],
                #         y=sub["sentiment_score"],
                #         name=dn(c),
                #         mode="lines+markers",
                #         line=dict(color=co_colors.get(c, "#1f77b4"), width=2),
                #         marker=dict(size=5),
                #     ))
                # fig_peer.update_layout(height=470, yaxis_title="Sentiment", xaxis_title="Period")
                # st.plotly_chart(fig_peer, use_container_width=True)

                # Dense sentiment filter for peer lines
                peer_plot_df = merged[
                    (merged["company_name"].isin(peer_companies))
                    & (merged["topic_label"] == peer_topic)
                    & (merged["finbert_label"].isin(_SENTIMENT_LABELS))
                ][["company_name", "year", "quarter", score_col]].copy()

                peer_plot_df["q_num"] = peer_plot_df["quarter"].map(
                    {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
                )

                peer_plot_df = peer_plot_df.groupby(
                    ["company_name", "year", "quarter", "q_num"], as_index=False
                )[score_col].mean()

                peer_plot_df["period_key"] = (
                    peer_plot_df["year"] * 10 + peer_plot_df["q_num"]
                )
                peer_plot_df["period"] = (
                    peer_plot_df["year"].astype(str) + " " + peer_plot_df["quarter"]
                )

                period_order = (
                    peer_plot_df[["period", "period_key"]]
                    .drop_duplicates()
                    .sort_values("period_key")["period"]
                    .tolist()
                )

                fig_peer = go.Figure()
                for c in peer_companies:
                    sub = peer_plot_df[peer_plot_df["company_name"] == c].sort_values(
                        "period_key"
                    )
                    if not len(sub):
                        continue
                    fig_peer.add_trace(
                        go.Scatter(
                            x=sub["period"],
                            y=sub[score_col],
                            name=dn(c),
                            mode="lines+markers",
                            line=dict(color=co_colors.get(c, "#1f77b4"), width=2),
                            marker=dict(size=5),
                            hovertemplate=(
                                f"<b>{dn(c)}</b><br>"
                                "%{x}<br>"
                                "Score: %{y:.2f}<extra></extra>"
                            ),
                        )
                    )

                fig_peer.update_layout(
                    title=dict(
                        text=f"Sentiment Trend — {dn(peer_topic)}",
                        font=dict(size=13, color="#374151"),
                        x=0,
                    ),
                    height=470,
                    yaxis_title=score_label,
                    xaxis_title="Period",
                    xaxis=dict(
                        type="category",
                        categoryorder="array",
                        categoryarray=period_order,
                    ),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                )
                fig_peer.add_hline(
                    y=0, line_dash="dot", line_color="#d1d5db", line_width=1
                )
                st.plotly_chart(fig_peer, use_container_width=True)

                # ── Latest ranking bar chart: avg pos / (n_pos + n_neg) ──────
                rank_df = ranked_company_pos_dense(
                    selected, sel_year, sel_q, topic=peer_topic
                )
                if len(rank_df):
                    # Compute all-companies avg(pos - neg) as a grey reference bar
                    _all_rank_df = ranked_company_pos_dense(
                        all_cos, sel_year, sel_q, topic=peer_topic
                    )
                    _all_avg_score = (
                        float(_all_rank_df["score"].mean())
                        if len(_all_rank_df)
                        else np.nan
                    )

                    rank_df = rank_df.sort_values("score", ascending=True)

                    # Append the all-companies avg row
                    if not np.isnan(_all_avg_score):
                        avg_row = pd.DataFrame(
                            [
                                {
                                    "company": "__all_avg__",
                                    "display": f"All cos avg (n={len(all_cos)})",
                                    "score": _all_avg_score,
                                }
                            ]
                        )
                        rank_df_plot = pd.concat([rank_df, avg_row], ignore_index=True)
                    else:
                        rank_df_plot = rank_df.copy()

                    bar_colors = [
                        "#9ca3af"
                        if c == "__all_avg__"
                        else ("#3fb950" if s >= 0 else "#f87171")
                        for c, s in zip(rank_df_plot["company"], rank_df_plot["score"])
                    ]

                    st.markdown(
                        f"<div style='font-size:0.7rem;font-weight:700;letter-spacing:0.09em;"
                        f"color:#6b7280;margin-top:20px;margin-bottom:4px'>"
                        f"📊 LATEST RANKING &nbsp;·&nbsp; {period_label.upper()} "
                        f"&nbsp;·&nbsp; AVG (POS − NEG) · DENSE SENTENCES</div>",
                        unsafe_allow_html=True,
                    )
                    fig_rank = go.Figure(
                        go.Bar(
                            x=rank_df_plot["score"],
                            y=rank_df_plot["display"],
                            orientation="h",
                            marker=dict(
                                color=bar_colors,
                                cornerradius=4,
                                line=dict(width=0),
                            ),
                            text=[f"{v:.2f}" for v in rank_df_plot["score"]],
                            textposition="outside",
                            hovertemplate="%{y}<br>Avg (pos − neg): %{x:.2f}<extra></extra>",
                        )
                    )
                    _x_min = min(rank_df_plot["score"].min(), 0) - 0.05
                    _x_max = rank_df_plot["score"].max() + 0.15
                    fig_rank.update_layout(
                        height=max(280, len(rank_df_plot) * 48 + 80),
                        xaxis=dict(
                            range=[_x_min, _x_max],
                            zeroline=False,
                            showgrid=False,
                        ),
                        yaxis=dict(autorange="reversed"),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        margin=dict(t=10, b=30, l=10, r=60),
                    )
                    st.plotly_chart(fig_rank, use_container_width=True)


# ── Tab 4: Single Filing Drill-Down ─────────────────────────────────────────
with tab4:
    st.subheader("Single Filing Drill-Down")

    fc1, fc2, fc3 = st.columns(3)
    _msft_default = next(
        (i for i, c in enumerate(all_cos) if "microsoft" in c.lower()), 0
    )
    f_co = fc1.selectbox("Company", all_cos, index=_msft_default, format_func=dn, key="t4_co")
    f_yr = fc2.selectbox("Year", list(reversed(avail_yrs)), key="t4_yr")
    f_q = fc3.selectbox("Quarter", ["Q4", "Q3", "Q2", "Q1"], key="t4_q")

    fd = merged[
        (merged["company_name"] == f_co)
        & (merged["year"] == f_yr)
        & (merged["quarter"] == f_q)
    ].copy()

    if not len(fd):
        st.info(f"No data for {dn(f_co)} {f_yr} {f_q}.")
    else:
        f_pos = float(fd["pos_prob"].mean())
        f_neu = float(fd["neu_prob"].mean())
        f_neg = float(fd["neg_prob"].mean())
        f_net = float(fd["sentiment_score"].mean()) if "sentiment_score" in fd.columns else f_pos - f_neg

        def _sent_label(s):
            if s >= 0.5:
                return "Strongly Positive"
            if s >= 0.1:
                return "Positive"
            if s > -0.1:
                return "Neutral"
            if s > -0.5:
                return "Negative"
            return "Strongly Negative"

        def _sent_color(s):
            if s >= 0.1:
                return "#16a34a"
            if s > -0.1:
                return "#6b7280"
            return "#dc2626"

        sent_label = _sent_label(f_net)
        sent_color = _sent_color(f_net)
        sign = "+" if f_net >= 0 else ""

        # ── Filing header + Overall Sentiment ────────────────────────────────
        hdr_left, hdr_right = st.columns([1.6, 1])
        with hdr_left:
            st.markdown(
                f"""
                <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;
                            padding:20px 24px 18px;box-shadow:0 1px 4px rgba(0,0,0,.05);">
                  <p style="margin:0 0 4px;font-size:11px;font-weight:600;color:#6b7280;
                             letter-spacing:.08em;text-transform:uppercase;">Selected Filing</p>
                  <p style="margin:0;font-size:24px;font-weight:700;color:#111827;">{dn(f_co)}</p>
                  <p style="margin:4px 0 0;font-size:12px;color:#6b7280;">
                    EDGAR SEC 10-Q &nbsp;·&nbsp; MD&amp;A section &nbsp;·&nbsp;
                    {f_yr} {f_q} &nbsp;·&nbsp; LDA + FinBERT pipeline
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with hdr_right:
            st.markdown(
                f"""
                <div style="background:#fff;border:1px solid #e5e7eb;border-radius:14px;
                            padding:20px 24px 18px;box-shadow:0 1px 4px rgba(0,0,0,.05);
                            text-align:right;">
                  <p style="margin:0 0 4px;font-size:11px;font-weight:600;color:#6b7280;
                             letter-spacing:.08em;text-transform:uppercase;">Overall Sentiment</p>
                  <p style="margin:0;font-size:42px;font-weight:800;color:{sent_color};
                             line-height:1.1;">{sign}{f_net:.2f}</p>
                  <p style="margin:6px 0 0;font-size:14px;font-weight:600;color:{sent_color};">
                    {sent_label}
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── 3 Metric cards ───────────────────────────────────────────────────
        mc1, mc2, mc3 = st.columns(3)
        _card_css = (
            "background:#fff;border:1px solid #e5e7eb;border-radius:14px;"
            "padding:18px 22px 16px;box-shadow:0 1px 4px rgba(0,0,0,.05);text-align:center;"
        )
        for col, label, value, color, icon in [
            (mc1, "Avg Positive Probability", f_pos, "#16a34a", "●"),
            (mc2, "Avg Negative Probability", f_neg, "#dc2626", "●"),
            (mc3, "Avg Neutral Probability",  f_neu, "#6b7280", "●"),
        ]:
            col.markdown(
                f"""
                <div style="{_card_css}">
                  <p style="margin:0 0 6px;font-size:11px;font-weight:600;color:#6b7280;
                             letter-spacing:.07em;text-transform:uppercase;">{label}</p>
                  <p style="margin:0;font-size:34px;font-weight:800;color:{color};line-height:1.1;">
                    {value:.3f}
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Risk / Opportunity signal cards ──────────────────────────────────
        topic_df = (
            fd.groupby("topic_label")[["pos_prob", "neg_prob", "sentiment_score"]]
            .mean()
            .reset_index()
            .rename(columns={"topic_label": "Topic", "pos_prob": "Avg pos", "neg_prob": "Avg neg", "sentiment_score": "Net"})
            .sort_values("Net", ascending=False)
        )

        if len(topic_df) >= 2:
            opp_topic  = topic_df.iloc[0]
            risk_topic = topic_df.iloc[-1]
            sig1, sig2 = st.columns(2)
            sig1.markdown(
                f"""
                <div style="background:#fff5f5;border:1px solid #fecaca;border-left:4px solid #dc2626;
                            border-radius:14px;padding:18px 22px;box-shadow:0 1px 4px rgba(0,0,0,.04);">
                  <p style="margin:0 0 8px;font-size:11px;font-weight:700;color:#dc2626;
                             letter-spacing:.08em;text-transform:uppercase;">● Top Risk Signal</p>
                  <p style="margin:0;font-size:14px;color:#374151;line-height:1.55;">
                    Topic <strong>{risk_topic['Topic']}</strong> has the lowest net sentiment
                    (<strong>{risk_topic['Net']:+.2f}</strong>) — highest negative exposure in this filing.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            sig2.markdown(
                f"""
                <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-left:4px solid #16a34a;
                            border-radius:14px;padding:18px 22px;box-shadow:0 1px 4px rgba(0,0,0,.04);">
                  <p style="margin:0 0 8px;font-size:11px;font-weight:700;color:#16a34a;
                             letter-spacing:.08em;text-transform:uppercase;">● Top Opportunity Signal</p>
                  <p style="margin:0;font-size:14px;color:#374151;line-height:1.55;">
                    Topic <strong>{opp_topic['Topic']}</strong> has the highest net sentiment
                    (<strong>{opp_topic['Net']:+.2f}</strong>) — strongest positive signal in this filing.
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Topics (LDA) card list + histogram ───────────────────────────────
        _TOPIC_COLORS = [
            "#16a34a","#f59e0b","#dc2626","#2563eb","#7c3aed",
            "#0891b2","#be185d","#65a30d","#ea580c","#6b7280",
        ]
        topic_color_map = {
            t: _TOPIC_COLORS[i % len(_TOPIC_COLORS)]
            for i, t in enumerate(topic_df.sort_values("Net", ascending=False)["Topic"])
        }

        t4_left, t4_right = st.columns([1, 1.6])
        with t4_left:
            st.markdown(
                "<p style='font-size:11px;font-weight:700;color:#6b7280;"
                "letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px;'>Topics (LDA)</p>",
                unsafe_allow_html=True,
            )
            for _, row in topic_df.iterrows():
                c = topic_color_map.get(row["Topic"], "#6b7280")
                net = row["Net"]
                sign_t = "+" if net >= 0 else ""
                net_color = "#16a34a" if net >= 0 else "#dc2626"
                st.markdown(
                    f"""
                    <div style="background:#fff;border:1px solid #e5e7eb;border-radius:10px;
                                padding:11px 16px;margin-bottom:8px;
                                box-shadow:0 1px 3px rgba(0,0,0,.04);">
                      <div style="display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:14px;font-weight:600;color:#111827;">
                          <span style="color:{c};margin-right:6px;">●</span>{row['Topic']}
                        </span>
                        <span style="font-size:16px;font-weight:700;color:{net_color};">
                          {sign_t}{net:.2f}
                        </span>
                      </div>
                      <div style="font-size:11px;color:#6b7280;margin-top:3px;padding-left:20px;">
                        pos {row['Avg pos']:.2f} &nbsp;/&nbsp; neg {row['Avg neg']:.2f}
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with t4_right:
            st.markdown(
                f"<p style='font-size:11px;font-weight:700;color:#6b7280;"
                f"letter-spacing:.08em;text-transform:uppercase;margin-bottom:10px;'>"
                f"{dn(f_co)} vs Corpus Average — Net Sentiment</p>",
                unsafe_allow_html=True,
            )
            # ── Net sentiment: selected company vs corpus average ─────────────
            _q_ord = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

            co_ts = (
                merged[merged["company_name"] == f_co]
                .groupby(["year", "quarter"])
                .apply(lambda g: float(g["pos_prob"].mean() - g["neg_prob"].mean()))
                .reset_index(name="net")
            )
            co_ts["period_key"] = co_ts["year"] * 10 + co_ts["quarter"].map(_q_ord)
            co_ts["period"] = co_ts["year"].astype(str) + " " + co_ts["quarter"]
            co_ts = co_ts.sort_values("period_key")

            avg_ts = (
                merged
                .groupby(["year", "quarter"])
                .apply(lambda g: float(g["pos_prob"].mean() - g["neg_prob"].mean()))
                .reset_index(name="net")
            )
            avg_ts["period_key"] = avg_ts["year"] * 10 + avg_ts["quarter"].map(_q_ord)
            avg_ts["period"] = avg_ts["year"].astype(str) + " " + avg_ts["quarter"]
            avg_ts = avg_ts.sort_values("period_key")

            # align to periods present in both so x-axis is consistent
            all_periods = avg_ts[["period", "period_key"]].drop_duplicates()  # type: ignore[union-attr]
            all_periods = all_periods.sort_values(by="period_key")  # type: ignore[union-attr]

            co_ts_full = all_periods.merge(co_ts[["period", "net"]], on="period", how="left")
            avg_ts_full = all_periods.merge(avg_ts[["period", "net"]], on="period", how="left")

            fig_cmp = go.Figure()
            fig_cmp.add_trace(go.Scatter(
                x=avg_ts_full["period"],
                y=avg_ts_full["net"],
                mode="lines",
                name="Corpus avg",
                line=dict(color="#9ca3af", width=2, dash="dot"),
                hovertemplate="%{x}<br>Avg: %{y:.3f}<extra></extra>",
            ))
            fig_cmp.add_trace(go.Scatter(
                x=co_ts_full["period"],
                y=co_ts_full["net"],
                mode="lines+markers",
                name=dn(f_co),
                line=dict(color="#16a34a", width=2.5),
                marker=dict(size=5, color="#16a34a"),
                hovertemplate="%{x}<br>" + dn(f_co) + ": %{y:.3f}<extra></extra>",
            ))

            # mark the selected quarter
            sel_period = f"{f_yr} {f_q}"
            sel_row = co_ts_full[co_ts_full["period"] == sel_period]
            if len(sel_row) and not pd.isna(sel_row["net"].iloc[0]):
                fig_cmp.add_trace(go.Scatter(
                    x=[sel_period],
                    y=[sel_row["net"].iloc[0]],
                    mode="markers",
                    name="Selected quarter",
                    marker=dict(color="#16a34a", size=13, symbol="circle",
                                line=dict(color="white", width=2)),
                    hovertemplate=f"{sel_period}<br>{dn(f_co)}: %{{y:.3f}}<extra></extra>",
                ))

            fig_cmp.add_hline(y=0, line_dash="solid", line_color="#e5e7eb", line_width=1)

            # show one tick label per year (Q1 of each year) to avoid crowding
            year_q1_periods = [
                p for p in co_ts_full["period"].tolist() if p.endswith("Q1")
            ]
            fig_cmp.update_layout(
                height=460,
                xaxis_title="Quarter",
                yaxis_title="Net sentiment (pos − neg)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis=dict(
                    tickmode="array",
                    tickvals=year_q1_periods,
                    ticktext=[p.replace(" Q1", "") for p in year_q1_periods],
                    tickangle=0,
                ),
            )
            st.plotly_chart(fig_cmp, use_container_width=True)


# ── Tab 5: Topic Modeling ────────────────────────────────────────────────────
with tab5:
    st.subheader("Topic Modeling Outputs")

    tm_c1, tm_c2 = st.columns([1, 1])
    tm_topic_mode = tm_c1.selectbox(
        "Topic list filter", ["All", "Top", "Bottom"], key="tm_topic_mode"
    )
    max_tm = max(1, min(15, len(all_topics)))
    tm_n_topics = safe_n_slider(tm_c2, "N topics", max_tm, 8, "tm_n_topics")
    # tm_n_topics   = tm_c2.slider("N topics", 1, max_tm, min(8, max_tm), key="tm_n_topics")

    tm_df = merged[merged["company_name"].isin(selected)].copy()
    if sel_year is not None:
        tm_df = tm_df[tm_df["year"] == sel_year]
    if sel_q is not None:
        tm_df = tm_df[tm_df["quarter"] == sel_q]

    if not len(tm_df):
        st.info("No topic data available for the selected filters.")
    else:
        topic_stats = (
            tm_df.groupby("topic_label")
            .agg(
                docs=("sentence_id", "count"),
                avg_pos=("pos_prob", "mean"),
                avg_neg=("neg_prob", "mean"),
                avg_topic_prob=("topic_prob", "mean"),
            )
            .reset_index()
            .rename(columns={"topic_label": "Topic"})
            .sort_values("docs", ascending=False)
        )

        if tm_topic_mode == "Top":
            topic_stats = topic_stats.head(tm_n_topics)
        elif tm_topic_mode == "Bottom":
            topic_stats = topic_stats.tail(tm_n_topics).sort_values("docs")

        if not len(topic_stats):
            st.info("No topics after filtering.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Topics shown", f"{len(topic_stats)}")
            m2.metric("Sentences in scope", f"{len(tm_df):,}")
            m3.metric("Companies in scope", f"{tm_df['company_name'].nunique()}")

            tm_bubble_col = score_col
            fig_bubble = go.Figure(
                go.Scatter(
                    x=topic_stats["docs"],
                    y=topic_stats["avg_pos"] - topic_stats["avg_neg"],
                    mode="markers",
                    text=topic_stats["Topic"],
                    customdata=np.stack(
                        [topic_stats["avg_pos"], topic_stats["avg_neg"]], axis=-1
                    ),
                    marker=dict(
                        size=np.clip(
                            topic_stats["avg_topic_prob"].fillna(0.3) * 65, 10, 60
                        ),
                        color=topic_stats["avg_pos"] - topic_stats["avg_neg"],
                        colorscale=[
                            [0.0, "#dc2626"],
                            [0.5, "#fef9c3"],
                            [1.0, "#16a34a"],
                        ],
                        cmid=0,
                        showscale=True,
                        colorbar=dict(title="Net sentiment"),
                        line=dict(width=1, color="#333"),
                        opacity=0.85,
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>Docs: %{x}<br>"
                        "Avg pos: %{customdata[0]:.3f}<br>Avg neg: %{customdata[1]:.3f}<extra></extra>"
                    ),
                )
            )
            fig_bubble.update_layout(
                height=460,
                xaxis_title="Topic prevalence (sentence count)",
                yaxis_title="Net sentiment (pos − neg)",
            )
            st.plotly_chart(fig_bubble, use_container_width=True)

            _tm_cols = st.columns([1.15, 1.25])
            left_tm, right_tm = _tm_cols[0], _tm_cols[1]

            with left_tm:
                st.write("Top words per topic")
                word_n = st.slider("Words per topic", 5, 20, 10, key="tm_word_n")
                kw_df = topic_keyword_table(
                    tm_df, topic_stats["Topic"].tolist(), top_n_words=word_n
                )

                # avg_pos: mean pos_prob over positive-labeled sentences only
                # avg_neg: mean neg_prob over negative-labeled sentences only
                _kw_pos = tm_df[tm_df["finbert_label"] == "positive"].groupby("topic_label")["pos_prob"].mean()
                _kw_pos = pd.DataFrame({"Topic": _kw_pos.index, "Avg pos": _kw_pos.values})
                _kw_neg = tm_df[tm_df["finbert_label"] == "negative"].groupby("topic_label")["neg_prob"].mean()
                _kw_neg = pd.DataFrame({"Topic": _kw_neg.index, "Avg neg": _kw_neg.values})
                kw_df = (
                    kw_df
                    .merge(topic_stats[["Topic", "docs"]].rename(columns={"docs": "Docs"}), on="Topic", how="left")
                    .merge(_kw_pos, on="Topic", how="left")
                    .merge(_kw_neg, on="Topic", how="left")
                )
                kw_df["Net"] = kw_df["Avg pos"].fillna(0) - kw_df["Avg neg"].fillna(0)
                kw_df = kw_df.sort_values("Net", ascending=False).drop(columns=["Net"])
                st.dataframe(kw_df, use_container_width=True, height=420)

            with right_tm:
                st.write("Company-topic distribution")
                max_dist = max(1, min(12, len(topic_stats)))
                top_topic_n = safe_n_slider(
                    st, "Topics in distribution", max_dist, 6, "tm_dist_n"
                )
                # top_topic_n = st.slider("Topics in distribution", 1, max_dist, min(6, max_dist), key="tm_dist_n")
                top_topics = topic_stats["Topic"].head(top_topic_n).tolist()
                dist = (
                    tm_df[tm_df["topic_label"].isin(top_topics)]
                    .groupby(["company_name", "topic_label"])["sentence_id"]
                    .count()
                    .reset_index(name="n")
                )
                if len(dist):
                    dist["share"] = (
                        dist["n"]
                        / dist.groupby("company_name")["n"].transform("sum")
                        * 100
                    )
                    fig_dist = go.Figure()
                    for t in top_topics:
                        sub = dist[dist["topic_label"] == t]
                        fig_dist.add_trace(
                            go.Bar(
                                x=[dn(c) for c in sub["company_name"]],
                                y=sub["share"],
                                name=dn(t),
                            )
                        )
                    fig_dist.update_layout(
                        barmode="stack", height=420, yaxis_title="Topic share (%)"
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                else:
                    st.info("No company-topic distribution for selected settings.")

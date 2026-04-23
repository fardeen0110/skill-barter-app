import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


st.set_page_config(page_title="Skill Barter Platform", page_icon="🤝", layout="wide")


SKILL_GROUPS = {
    "Python": "Technology",
    "Java": "Technology",
    "Web Development": "Technology",
    "Data Science": "Technology",
    "UI/UX Design": "Design",
    "Graphic Design": "Design",
    "Photography": "Creative",
    "Video Editing": "Creative",
    "Public Speaking": "Communication",
    "Content Writing": "Communication",
    "Digital Marketing": "Business",
    "Financial Planning": "Business",
    "Guitar": "Music",
    "Piano": "Music",
    "Fitness Coaching": "Wellness",
    "Yoga": "Wellness",
}

SKILLS = list(SKILL_GROUPS.keys())
INTERACTION_LABELS = {0: "Flexible", 1: "Online", 2: "In Person"}
AVAILABILITY_LABELS = {
    0.15: "Weekday mornings",
    0.35: "Weekday evenings",
    0.55: "Weekend mornings",
    0.75: "Weekend afternoons",
    0.90: "Late evenings",
}

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Ishaan", "Arjun", "Kabir", "Rohan", "Dev", "Kunal", "Rahul",
    "Ananya", "Diya", "Saanvi", "Myra", "Aadhya", "Kiara", "Meera", "Riya", "Sara", "Ira",
    "Neha", "Priya", "Nisha", "Siddharth", "Tanvi", "Varun", "Nikhil", "Aanya", "Zara", "Reyansh",
    "Krishna", "Yash", "Aman", "Pooja", "Sneha", "Maya", "Aisha", "Aria", "Ritika", "Shreya",
]

LAST_NAMES = [
    "Sharma", "Patel", "Gupta", "Verma", "Mehta", "Reddy", "Kapoor", "Joshi", "Nair", "Malhotra",
    "Iyer", "Bose", "Chopra", "Khanna", "Singh", "Saxena", "Menon", "Desai", "Bhat", "Arora",
]


def initialize_session_state():
    defaults = {
        "users": {
            "demo": {"password": "demo123"}
        },
        "profiles": {
            "demo": {
                "offer_skill": "Python",
                "learn_skill": "UI/UX Design",
                "experience_level": 0.70,
                "availability_value": 0.55,
                "availability_label": AVAILABILITY_LABELS[0.55],
                "rating_avg": 4.4,
                "interaction_preference_value": 1.0,
                "interaction_preference_label": "Online",
                "prior_exchange_count": 2,
            }
        },
        "authenticated_user": None,
        "auth_mode": "Login",
        "last_matches": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, float(value)))


def skill_similarity(skill_a, skill_b):
    if skill_a == skill_b:
        return 1.0
    if SKILL_GROUPS[skill_a] == SKILL_GROUPS[skill_b]:
        return 0.72
    return 0.25


def availability_overlap(slot_a, slot_b):
    distance = abs(slot_a - slot_b)
    return clamp(1.0 - distance / 0.8)


def interaction_compatibility(pref_a, pref_b):
    if pref_a == pref_b:
        return 1.0
    if pref_a == 0 or pref_b == 0:
        return 0.75
    return 0.45


def generate_match_probability(skill_sim, availability, experience_gap, rating_avg, interaction_pref):
    probability = (
        0.35 * skill_sim
        + 0.25 * availability
        + 0.20 * (1.0 - experience_gap)
        + 0.15 * (rating_avg / 5.0)
        + 0.05 * interaction_pref
    )
    return np.clip(probability, 0, 1)


@st.cache_data
def load_dataset():
    rng = np.random.default_rng(42)
    sample_size = 1000

    skill_similarity_values = np.clip(
        np.concatenate(
            [
                rng.beta(5, 2, size=int(sample_size * 0.45)),
                rng.beta(2, 5, size=int(sample_size * 0.35)),
                rng.beta(3, 3, size=sample_size - int(sample_size * 0.45) - int(sample_size * 0.35)),
            ]
        ),
        0,
        1,
    )
    rng.shuffle(skill_similarity_values)

    availability_values = np.clip(rng.beta(3.8, 2.4, sample_size), 0, 1)
    experience_gap_values = np.clip(rng.beta(2.1, 3.2, sample_size), 0, 1)
    rating_values = np.clip(rng.normal(4.2, 0.45, sample_size), 2.5, 5.0)
    interaction_values = rng.choice([0.3, 0.65, 1.0], size=sample_size, p=[0.18, 0.32, 0.50])
    prior_exchange_values = np.clip(rng.poisson(2.5, sample_size), 0, 10)

    probabilities = generate_match_probability(
        skill_similarity_values,
        availability_values,
        experience_gap_values,
        rating_values,
        interaction_values,
    )

    thresholds = rng.normal(0.63, 0.05, sample_size)
    match_success = (probabilities >= thresholds).astype(int)

    dataset = pd.DataFrame(
        {
            "SkillSimilarity": skill_similarity_values,
            "AvailabilityOverlap": availability_values,
            "ExperienceGap": experience_gap_values,
            "UserRatingAvg": rating_values,
            "InteractionPreference": interaction_values,
            "PriorExchangeCount": prior_exchange_values,
            "MatchSuccess": match_success,
        }
    )
    return dataset


@st.cache_resource
def train_model():
    dataset = load_dataset()
    feature_columns = [
        "SkillSimilarity",
        "AvailabilityOverlap",
        "ExperienceGap",
        "UserRatingAvg",
        "InteractionPreference",
        "PriorExchangeCount",
    ]

    X = dataset[feature_columns]
    y = dataset["MatchSuccess"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    logistic_model = LogisticRegression(max_iter=1000, random_state=42)
    logistic_model.fit(X_train, y_train)
    logistic_accuracy = accuracy_score(y_test, logistic_model.predict(X_test))

    random_forest_model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_split=6,
        min_samples_leaf=3,
        random_state=42,
    )
    random_forest_model.fit(X_train, y_train)
    random_forest_accuracy = accuracy_score(y_test, random_forest_model.predict(X_test))

    metrics = {
        "logistic_accuracy": logistic_accuracy,
        "random_forest_accuracy": random_forest_accuracy,
        "feature_columns": feature_columns,
    }
    return random_forest_model, logistic_model, metrics


@st.cache_data
def generate_community_users(total_users=36):
    rng = np.random.default_rng(7)
    full_names = []
    while len(full_names) < total_users:
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if name not in full_names:
            full_names.append(name)

    availability_choices = list(AVAILABILITY_LABELS.keys())

    users = []
    for name in full_names:
        offer_skill = rng.choice(SKILLS)
        learn_candidates = [skill for skill in SKILLS if skill != offer_skill]
        learn_skill = rng.choice(learn_candidates)
        availability_value = float(rng.choice(availability_choices))
        interaction_code = int(rng.choice([0, 1, 2], p=[0.20, 0.50, 0.30]))
        rating_avg = float(np.clip(rng.normal(4.1, 0.45), 3.0, 5.0))
        experience_level = float(np.clip(rng.beta(3.2, 2.0), 0.15, 0.98))
        prior_exchange_count = int(np.clip(rng.poisson(3.0), 0, 12))

        users.append(
            {
                "name": name,
                "offer_skill": offer_skill,
                "learn_skill": learn_skill,
                "availability_value": availability_value,
                "availability_label": AVAILABILITY_LABELS[availability_value],
                "experience_level": experience_level,
                "rating_avg": rating_avg,
                "interaction_preference_value": float(interaction_code),
                "interaction_preference_label": INTERACTION_LABELS[interaction_code],
                "prior_exchange_count": prior_exchange_count,
            }
        )

    return pd.DataFrame(users)


def build_feature_vector(user_profile, candidate_profile):
    skill_sim = skill_similarity(user_profile["learn_skill"], candidate_profile["offer_skill"])
    availability = availability_overlap(
        user_profile["availability_value"],
        candidate_profile["availability_value"],
    )
    experience_gap = abs(user_profile["experience_level"] - candidate_profile["experience_level"])
    rating_avg = (user_profile["rating_avg"] + candidate_profile["rating_avg"]) / 2.0
    interaction_pref = interaction_compatibility(
        user_profile["interaction_preference_value"],
        candidate_profile["interaction_preference_value"],
    )
    prior_exchange_count = min(
        (user_profile["prior_exchange_count"] + candidate_profile["prior_exchange_count"]) / 2.0,
        10.0,
    )

    return {
        "SkillSimilarity": round(skill_sim, 4),
        "AvailabilityOverlap": round(availability, 4),
        "ExperienceGap": round(experience_gap, 4),
        "UserRatingAvg": round(rating_avg, 4),
        "InteractionPreference": round(interaction_pref, 4),
        "PriorExchangeCount": round(prior_exchange_count, 4),
    }


def get_user_profile(username):
    return st.session_state["profiles"].get(username)


def find_matches(user_profile):
    model, _, metrics = train_model()
    community_users = generate_community_users()

    feature_rows = []
    candidate_rows = []

    for _, candidate in community_users.iterrows():
        candidate_profile = candidate.to_dict()
        if candidate_profile["offer_skill"] == user_profile["offer_skill"]:
            continue

        features = build_feature_vector(user_profile, candidate_profile)
        feature_rows.append(features)
        candidate_rows.append(candidate_profile)

    if not feature_rows:
        return []

    features_df = pd.DataFrame(feature_rows)[metrics["feature_columns"]]
    probabilities = model.predict_proba(features_df)[:, 1]

    results = []
    for candidate, probability, feature_set in zip(candidate_rows, probabilities, feature_rows):
        if feature_set["SkillSimilarity"] < 0.25:
            continue

        results.append(
            {
                "name": candidate["name"],
                "offer_skill": candidate["offer_skill"],
                "learn_skill": candidate["learn_skill"],
                "match_score": int(round(probability * 100)),
                "availability_label": candidate["availability_label"],
                "interaction_label": candidate["interaction_preference_label"],
            }
        )

    results.sort(key=lambda item: item["match_score"], reverse=True)
    return results[:5]


def render_styles():
    st.markdown(
        """
        <style>
        :root {
            --bg-1: #050816;
            --bg-2: #0b1120;
            --panel: rgba(15, 23, 42, 0.82);
            --panel-strong: rgba(17, 24, 39, 0.94);
            --line: rgba(148, 163, 184, 0.16);
            --line-strong: rgba(148, 163, 184, 0.28);
            --text: #f8fafc;
            --muted: #94a3b8;
            --soft: #cbd5e1;
            --cyan: #67e8f9;
            --blue: #38bdf8;
            --mint: #34d399;
            --glow: rgba(56, 189, 248, 0.18);
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.16), transparent 24%),
                radial-gradient(circle at 85% 12%, rgba(52, 211, 153, 0.10), transparent 22%),
                radial-gradient(circle at 50% 100%, rgba(99, 102, 241, 0.08), transparent 30%),
                linear-gradient(145deg, var(--bg-1) 0%, var(--bg-2) 48%, #111827 100%);
            color: var(--text);
        }
        [data-testid="stAppViewContainer"] > .main {
            padding-top: 1.2rem;
        }
        [data-testid="block-container"] {
            padding-top: 1rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }
        [data-testid="stSidebar"] {
            background:
                radial-gradient(circle at top, rgba(56, 189, 248, 0.12), transparent 26%),
                linear-gradient(180deg, #08101d 0%, #0f172a 100%);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] * {
            color: #e5eefb !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: var(--text);
            letter-spacing: -0.02em;
        }
        p, label, .stMarkdown, .stCaption {
            color: var(--soft);
        }
        .hero-card, .glass-card, .match-card, .focus-card {
            position: relative;
            overflow: hidden;
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.92) 0%, rgba(15, 23, 42, 0.74) 100%);
            border: 1px solid var(--line);
            border-radius: 24px;
            box-shadow:
                0 24px 60px rgba(0, 0, 0, 0.32),
                inset 0 1px 0 rgba(255, 255, 255, 0.04);
            backdrop-filter: blur(14px);
        }
        .hero-card::before, .glass-card::before, .match-card::before, .focus-card::before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(135deg, rgba(103, 232, 249, 0.10), transparent 36%, transparent 65%, rgba(52, 211, 153, 0.08));
            pointer-events: none;
        }
        .hero-card {
            padding: 1.9rem 2rem;
            margin-bottom: 1.25rem;
        }
        .glass-card {
            padding: 1.35rem;
            min-height: 156px;
        }
        .match-card {
            padding: 1.35rem 1.4rem;
            margin-bottom: 1rem;
            transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }
        .match-card:hover {
            transform: translateY(-2px);
            border-color: rgba(103, 232, 249, 0.28);
            box-shadow: 0 26px 60px rgba(0, 0, 0, 0.34), 0 0 0 1px rgba(103, 232, 249, 0.06);
        }
        .focus-card {
            padding: 1.1rem 1.2rem;
            border-radius: 18px;
            min-height: 100px;
        }
        .kicker {
            color: var(--cyan);
            text-transform: uppercase;
            letter-spacing: 0.16em;
            font-size: 0.75rem;
            font-weight: 800;
        }
        .hero-title {
            font-size: 2.45rem;
            line-height: 1.05;
            font-weight: 800;
            margin-top: 0.55rem;
            margin-bottom: 0.65rem;
            color: var(--text);
            max-width: 12ch;
        }
        .hero-subtitle {
            color: var(--soft);
            font-size: 1.02rem;
            line-height: 1.75;
            max-width: 64ch;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 0.55rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .metric-value {
            color: var(--text);
            font-size: 2.15rem;
            font-weight: 800;
            line-height: 1;
            margin-bottom: 0.6rem;
        }
        .section-title {
            color: var(--text);
            font-size: 1.3rem;
            font-weight: 700;
            margin: 1.1rem 0 0.8rem 0;
        }
        .muted-text {
            color: var(--soft);
            font-size: 0.97rem;
            line-height: 1.65;
        }
        .score-pill {
            display: inline-block;
            padding: 0.5rem 0.9rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(52, 211, 153, 0.18), rgba(56, 189, 248, 0.16));
            color: #d1fae5;
            font-weight: 800;
            font-size: 0.92rem;
            border: 1px solid rgba(103, 232, 249, 0.16);
            box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.06);
        }
        .tag {
            display: inline-block;
            margin-right: 0.5rem;
            margin-top: 0.75rem;
            padding: 0.34rem 0.78rem;
            border-radius: 999px;
            background: rgba(56, 189, 248, 0.10);
            color: #d6f4ff;
            font-size: 0.84rem;
            border: 1px solid rgba(56, 189, 248, 0.12);
        }
        .auth-shell {
            max-width: 560px;
            margin: 3.6rem auto 0 auto;
        }
        .auth-card {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.94), rgba(17, 24, 39, 0.86));
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1.15rem;
            box-shadow: 0 24px 56px rgba(0, 0, 0, 0.30);
        }
        .auth-divider {
            height: 1px;
            width: 100%;
            background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.28), transparent);
            margin: 1rem 0 0.6rem 0;
        }
        .stButton > button {
            width: 100%;
            min-height: 2.9rem;
            border-radius: 14px;
            border: 1px solid rgba(103, 232, 249, 0.16);
            background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
            color: white;
            font-weight: 800;
            box-shadow: 0 14px 30px rgba(14, 165, 233, 0.18);
        }
        .stButton > button:hover {
            border-color: rgba(103, 232, 249, 0.28);
            box-shadow: 0 18px 34px rgba(14, 165, 233, 0.24);
        }
        .stTextInput input, .stSelectbox div[data-baseweb="select"], .stSlider, .stNumberInput input {
            border-radius: 14px !important;
        }
        .stTextInput input, .stNumberInput input {
            background: rgba(15, 23, 42, 0.72) !important;
            border: 1px solid var(--line) !important;
            color: var(--text) !important;
        }
        div[data-baseweb="select"] > div {
            background: rgba(15, 23, 42, 0.72) !important;
            border: 1px solid var(--line) !important;
        }
        .stRadio > div {
            gap: 0.5rem;
        }
        .stAlert {
            border-radius: 18px;
            border: 1px solid var(--line-strong);
        }
        [data-testid="metric-container"] {
            background: transparent;
        }
        @media (max-width: 900px) {
            .hero-title {
                font-size: 2rem;
                max-width: 100%;
            }
            [data-testid="block-container"] {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_auth_page():
    st.markdown("<div class='auth-shell'>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-card auth-card">
            <div class="kicker">Skill Barter Platform</div>
            <div class="hero-title">Exchange what you know. Learn what you love.</div>
            <div class="hero-subtitle">
                Create a profile, explore compatible learners and mentors, and discover top skill matches powered quietly in the background by a smart matchmaking engine.
            </div>
            <div class="auth-divider"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    auth_mode = st.radio(
        "Choose an option",
        ["Login", "Signup"],
        index=0 if st.session_state["auth_mode"] == "Login" else 1,
        horizontal=True,
    )
    st.session_state["auth_mode"] = auth_mode

    with st.form("auth_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        confirm_password = ""
        if auth_mode == "Signup":
            confirm_password = st.text_input("Confirm password", type="password")
        submitted = st.form_submit_button(auth_mode)

    if submitted:
        username = username.strip()
        if not username or not password:
            st.error("Please enter both username and password.")
        elif auth_mode == "Signup":
            if username in st.session_state["users"]:
                st.error("That username already exists. Please choose another one.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                st.session_state["users"][username] = {"password": password}
                st.session_state["authenticated_user"] = username
                st.success("Account created successfully.")
                st.rerun()
        else:
            user_record = st.session_state["users"].get(username)
            if user_record and user_record["password"] == password:
                st.session_state["authenticated_user"] = username
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    st.caption("Demo account: username `demo`, password `demo123`")
    st.markdown("</div>", unsafe_allow_html=True)


def render_sidebar():
    current_user = st.session_state["authenticated_user"]
    st.sidebar.markdown("## Skill Barter")
    st.sidebar.caption(f"Signed in as {current_user}")
    page = st.sidebar.radio("Navigation", ["Dashboard", "Profile", "Matches"])

    if st.sidebar.button("Log out"):
        st.session_state["authenticated_user"] = None
        st.session_state["last_matches"] = []
        st.rerun()
    return page


def render_dashboard():
    current_user = st.session_state["authenticated_user"]
    profile = get_user_profile(current_user)
    community_users = generate_community_users()

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="kicker">Dashboard</div>
            <div class="hero-title">Welcome back, {current_user}</div>
            <div class="hero-subtitle">
                Build your barter profile, discover people with complementary skills, and start meaningful learning exchanges with the strongest matches first.
            </div>
            <div style="margin-top:1rem;">
                <span class="tag">Curated skill network</span>
                <span class="tag">Fast matching</span>
                <span class="tag">Private profile flow</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Community members</div>
                <div class="metric-value">{len(community_users)}</div>
                <div class="muted-text">Active simulated users ready for matching</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Skills available</div>
                <div class="metric-value">{len(SKILLS)}</div>
                <div class="muted-text">Across technology, design, business, wellness, and more</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        profile_status = "Ready" if profile else "Incomplete"
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="metric-label">Profile status</div>
                <div class="metric-value">{profile_status}</div>
                <div class="muted-text">Complete your profile to unlock the best recommendations</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div class='section-title'>Your Current Focus</div>", unsafe_allow_html=True)
    if profile:
        left, right = st.columns(2)
        with left:
            st.markdown(
                f"""
                <div class="focus-card">
                    <div class="metric-label">You Offer</div>
                    <div style="font-size:1.25rem; font-weight:700; color:#f8fafc;">{profile['offer_skill']}</div>
                    <div class="muted-text">This is the skill you bring into the exchange.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                f"""
                <div class="focus-card">
                    <div class="metric-label">You Want To Learn</div>
                    <div style="font-size:1.25rem; font-weight:700; color:#f8fafc;">{profile['learn_skill']}</div>
                    <div class="muted-text">We’ll prioritize members who can teach this well.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.warning("You have not created your profile yet. Visit the Profile page to get started.")


def render_profile():
    current_user = st.session_state["authenticated_user"]
    existing_profile = get_user_profile(current_user) or {}

    st.markdown(
        """
        <div class="hero-card">
            <div class="kicker">Profile</div>
            <div class="hero-title">Tell the platform what you can teach and what you want to learn</div>
            <div class="hero-subtitle">
                The app uses your preferences behind the scenes to estimate compatibility and surface the most promising barter partners.
            </div>
            <div style="margin-top:1rem;">
                <span class="tag">Simple setup</span>
                <span class="tag">No technical fields</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("profile_form"):
        offer_skill = st.selectbox(
            "Skill you can offer",
            SKILLS,
            index=SKILLS.index(existing_profile.get("offer_skill", SKILLS[0])),
        )

        learn_options = [skill for skill in SKILLS if skill != offer_skill]
        default_learn_skill = existing_profile.get("learn_skill", learn_options[0])
        if default_learn_skill not in learn_options:
            default_learn_skill = learn_options[0]
        learn_skill = st.selectbox(
            "Skill you want to learn",
            learn_options,
            index=learn_options.index(default_learn_skill),
        )

        experience_level = st.slider(
            "Your current confidence in the skill you offer",
            min_value=0.0,
            max_value=1.0,
            value=float(existing_profile.get("experience_level", 0.6)),
            step=0.05,
            help="Higher values mean you feel more comfortable teaching your offered skill.",
        )

        availability_label = st.selectbox(
            "Preferred availability",
            list(AVAILABILITY_LABELS.values()),
            index=list(AVAILABILITY_LABELS.values()).index(
                existing_profile.get("availability_label", AVAILABILITY_LABELS[0.55])
            ),
        )

        rating_avg = st.slider(
            "Your average collaboration rating",
            min_value=3.0,
            max_value=5.0,
            value=float(existing_profile.get("rating_avg", 4.3)),
            step=0.1,
        )

        interaction_label = st.selectbox(
            "Preferred interaction style",
            list(INTERACTION_LABELS.values()),
            index=list(INTERACTION_LABELS.values()).index(
                existing_profile.get("interaction_preference_label", "Online")
            ),
        )

        prior_exchange_count = st.slider(
            "Number of prior skill exchanges",
            min_value=0,
            max_value=10,
            value=int(existing_profile.get("prior_exchange_count", 1)),
            step=1,
        )

        save_profile = st.form_submit_button("Save Profile")

    if save_profile:
        availability_value = next(
            value for value, label in AVAILABILITY_LABELS.items() if label == availability_label
        )
        interaction_value = next(
            float(key) for key, label in INTERACTION_LABELS.items() if label == interaction_label
        )
        st.session_state["profiles"][current_user] = {
            "offer_skill": offer_skill,
            "learn_skill": learn_skill,
            "experience_level": experience_level,
            "availability_value": availability_value,
            "availability_label": availability_label,
            "rating_avg": rating_avg,
            "interaction_preference_value": interaction_value,
            "interaction_preference_label": interaction_label,
            "prior_exchange_count": prior_exchange_count,
        }
        st.success("Your profile has been saved.")


def render_match_card(match):
    st.markdown(
        f"""
        <div class="match-card">
            <div style="display:flex; justify-content:space-between; align-items:center; gap:1rem;">
                <div>
                    <div style="font-size:1.2rem; font-weight:700; color:#f8fafc;">{match["name"]}</div>
                    <div class="muted-text">Offers {match["offer_skill"]} and wants to learn {match["learn_skill"]}</div>
                </div>
                <div class="score-pill">{match["match_score"]}% match</div>
            </div>
            <div>
                <span class="tag">{match["availability_label"]}</span>
                <span class="tag">{match["interaction_label"]}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_matches():
    current_user = st.session_state["authenticated_user"]
    profile = get_user_profile(current_user)

    st.markdown(
        """
        <div class="hero-card">
            <div class="kicker">Matches</div>
            <div class="hero-title">Your best skill barter partners</div>
            <div class="hero-subtitle">
                Find learners and mentors whose skills and preferences align closely with your profile, then start with the strongest opportunities first.
            </div>
            <div style="margin-top:1rem;">
                <span class="tag">Top 5 ranked</span>
                <span class="tag">Compatibility score</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not profile:
        st.warning("Please complete your profile before searching for matches.")
        return

    col1, col2 = st.columns([1.4, 1.0])
    with col1:
        st.markdown(
            f"""
            <div class="glass-card">
                <div class="section-title">Your search setup</div>
                <div class="muted-text">Offering: <strong>{profile["offer_skill"]}</strong></div>
                <div class="muted-text">Learning: <strong>{profile["learn_skill"]}</strong></div>
                <div class="muted-text">Availability: <strong>{profile["availability_label"]}</strong></div>
                <div class="muted-text">Interaction: <strong>{profile["interaction_preference_label"]}</strong></div>
                <div style="margin-top:0.85rem;">
                    <span class="tag">{profile["offer_skill"]}</span>
                    <span class="tag">{profile["learn_skill"]}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        if st.button("Find Top 5 Matches"):
            st.session_state["last_matches"] = find_matches(profile)

    if st.session_state["last_matches"]:
        st.markdown("<div class='section-title'>Recommended matches</div>", unsafe_allow_html=True)
        for match in st.session_state["last_matches"]:
            render_match_card(match)
    else:
        st.info("Click the button above to generate your top 5 matches.")


def main():
    initialize_session_state()
    render_styles()
    train_model()

    if not st.session_state["authenticated_user"]:
        render_auth_page()
        return

    page = render_sidebar()
    if page == "Dashboard":
        render_dashboard()
    elif page == "Profile":
        render_profile()
    else:
        render_matches()


if __name__ == "__main__":
    main()

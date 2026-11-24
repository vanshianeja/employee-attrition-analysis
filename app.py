"""Employee Attrition Analysis - Streamlit dashboard (CSET211 final project)."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from cleaning import clean_data, corrupt_data, quality_report

DATA_PATH = "WA_Fn-UseC_-HR-Employee-Attrition.csv"
USELESS_COLS = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]
CLUSTER_COLS = ["Age", "MonthlyIncome", "TotalWorkingYears", "YearsAtCompany",
                "YearsSinceLastPromotion", "JobLevel", "JobSatisfaction",
                "WorkLifeBalance", "EnvironmentSatisfaction",
                "DistanceFromHome", "OverTime_Yes"]

PRIMARY, RED, GREEN, AMBER, SLATE = "#6366F1", "#F87171", "#34D399", "#FBBF24", "#94A3B8"

st.set_page_config(page_title="Employee Attrition Analysis", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1250px;}
.hero {background: linear-gradient(135deg, #4338CA 0%, #7C3AED 100%); color: white;
       padding: 28px 32px; border-radius: 18px; margin-bottom: 22px;
       box-shadow: 0 8px 30px rgba(99, 102, 241, 0.25);}
.hero h1 {color: white; margin: 0 0 6px 0; font-size: 2.1rem;}
.hero p {color: #E0E7FF; margin: 0; font-size: 1.02rem;}
.kpi {background: #151B2E; border-radius: 14px; padding: 16px 18px; border-left: 6px solid #6366F1;
      border-top: 1px solid #232B45; border-right: 1px solid #232B45; border-bottom: 1px solid #232B45;
      min-height: 135px; box-shadow: 0 4px 16px rgba(0, 0, 0, 0.35);}
.kpi .label {color: #94A3B8; font-size: 0.78rem; text-transform: uppercase; letter-spacing: .06em;}
.kpi .value {color: #F8FAFC; font-size: 1.55rem; font-weight: 700; line-height: 1.3; margin: 2px 0;}
.kpi .sub {color: #94A3B8; font-size: 0.85rem;}
.kpi.red {border-left-color: #F87171;} .kpi.green {border-left-color: #34D399;} .kpi.amber {border-left-color: #FBBF24;}
.insight {background: #1A2140; border: 1px solid #2B3563; border-radius: 12px; padding: 14px 16px;
          color: #E2E8F0; min-height: 120px;}
.insight b {color: #A5B4FC;}
.badge {display: inline-block; padding: 6px 18px; border-radius: 999px; color: #0B1020; font-weight: 800;}
.section-title {font-size: 1.15rem; font-weight: 700; margin: 6px 0 8px 0; color: #F1F5F9;}
</style>
""", unsafe_allow_html=True)


# ---------- helper functions ----------
def kpi(col, label, value, sub="", color=""):
    col.markdown(f"""<div class="kpi {color}"><div class="label">{label}</div>
                 <div class="value">{value}</div><div class="sub">{sub}</div></div>""",
                 unsafe_allow_html=True)


def style_fig(fig, height=380):
    fig.update_layout(template="plotly_dark", height=height, margin=dict(l=10, r=10, t=45, b=10),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="sans-serif", size=13, color="#E2E8F0"), title_font_size=16)
    return fig


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def prepare(df):
    """Raw dataframe -> (features X, target y). Same steps as the notebook."""
    y = (df["Attrition"] == "Yes").astype(int)
    X = pd.get_dummies(df.drop(columns=["Attrition"] + [c for c in USELESS_COLS if c in df.columns]),
                       drop_first=True)
    return X, y


def get_models():
    return {
        "Logistic Regression": (LogisticRegression(class_weight="balanced", max_iter=1000), True),
        "KNN (K=7)": (KNeighborsClassifier(n_neighbors=7), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42), False),
        "Random Forest": (RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42), False),
        "Naive Bayes": (GaussianNB(), False),
        "SVM (RBF kernel)": (SVC(kernel="rbf", class_weight="balanced"), True),
    }


@st.cache_data
def run_cross_validation():
    X, y = prepare(load_data())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, (model, needs_scaling) in get_models().items():
        est = make_pipeline(StandardScaler(), model) if needs_scaling else model
        s = cross_validate(est, X, y, cv=cv, scoring=["accuracy", "precision", "recall", "f1"])
        rows.append({"Model": name, "Accuracy": s["test_accuracy"].mean(),
                     "Precision": s["test_precision"].mean(), "Recall": s["test_recall"].mean(),
                     "F1": s["test_f1"].mean()})
    return pd.DataFrame(rows).sort_values("F1", ascending=False).round(3).reset_index(drop=True)


@st.cache_resource
def train_final_model():
    """Logistic Regression trained on all data - used for the Predict page."""
    X, y = prepare(load_data())
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(class_weight="balanced", max_iter=1000).fit(scaler.transform(X), y)
    return model, scaler, list(X.columns)


@st.cache_data
def run_segmentation(k=5):
    X, y = prepare(load_data())
    Xc = StandardScaler().fit_transform(X[CLUSTER_COLS].astype(float))
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xc)
    seg = X[CLUSTER_COLS].copy()
    seg["Cluster"] = km.labels_
    seg["Attrition"] = y.values
    summary = seg.groupby("Cluster").agg(Size=("Attrition", "size"), AttritionRate=("Attrition", "mean")).round(3)
    profile = seg.groupby("Cluster")[CLUSTER_COLS].mean().round(2)
    return seg, summary, profile


def lr_cv_f1(data):
    """Cross-validated F1 of Logistic Regression on a dataframe (used in the messy-data demo)."""
    X, y = prepare(data)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    est = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))
    return cross_val_score(est, X, y, cv=cv, scoring="f1").mean()


# ---------- sidebar ----------
PAGES = {"🏠  Overview": "overview", "🔎  Explore Data": "explore", "🏆  Model Comparison": "models",
         "👥  Employee Segments": "segments", "🎯  Predict Attrition Risk": "predict",
         "💡  Insights & Recommendations": "insights", "🧹  Data Quality & Cleaning": "quality"}
st.sidebar.markdown("## 📊 Attrition Analysis")
page = PAGES[st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")]
st.sidebar.divider()
st.sidebar.markdown("**Dataset**  \nIBM HR Analytics")
st.sidebar.markdown("**Final model**  \nLogistic Regression")
st.sidebar.markdown("**Segments**  \nK-Means (K=5)")
st.sidebar.divider()
st.sidebar.caption("CSET211 Statistical Machine Learning  \nFinal Project")

df = load_data()
X_all, y_all = prepare(df)
base = y_all.mean()


def hero(title, subtitle):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


# ---------- pages ----------
if page == "overview":
    hero("Employee Attrition Analysis",
         "Which employees leave, which groups are at risk, and how well machine learning can tell. "
         "Built with Logistic Regression, SVM, Random Forest and K-Means.")

    ot = df.assign(flag=y_all).groupby("OverTime")["flag"].mean()
    role = df.assign(flag=y_all).groupby("JobRole")["flag"].mean().sort_values(ascending=False)
    c1, c2, c3, c4 = st.columns(4)
    kpi(c1, "Employees", f"{len(df):,}", f"{df.shape[1]} columns")
    kpi(c2, "Attrition rate", f"{base * 100:.1f}%", f"{int(y_all.sum())} employees left", "red")
    kpi(c3, "Overtime effect", f"{ot['Yes'] / ot['No']:.1f}x",
        f"{ot['Yes'] * 100:.0f}% vs {ot['No'] * 100:.0f}% leave", "amber")
    kpi(c4, "Highest-risk role", role.index[0], f"{role.iloc[0] * 100:.0f}% leave", "red")

    st.write("")
    left, right = st.columns([1, 1.4])
    with left:
        fig = go.Figure(go.Pie(labels=["Stayed", "Left"], values=[(1 - base) * 100, base * 100], hole=0.62,
                               marker_colors=[PRIMARY, RED], textinfo="label+percent"))
        fig.update_layout(title="Who stays and who leaves", showlegend=False)
        st.plotly_chart(style_fig(fig, 340))
    with right:
        top = (role.head(6) * 100).round(1).sort_values()
        fig = px.bar(x=top.values, y=top.index, orientation="h", title="Attrition rate by job role (top 6)",
                     labels={"x": "Attrition rate (%)", "y": ""}, color=top.values,
                     color_continuous_scale=["#3730A3", RED])
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(fig, 340))

    st.markdown('<div class="section-title">Key findings</div>', unsafe_allow_html=True)
    ms = df.assign(flag=y_all).groupby("MaritalStatus")["flag"].mean().sort_values(ascending=False)
    i1, i2, i3 = st.columns(3)
    i1.markdown(f'<div class="insight"><b>Overtime matters.</b><br>{ot["Yes"] * 100:.1f}% of employees '
                f'working overtime left, against {ot["No"] * 100:.1f}% of the rest.</div>', unsafe_allow_html=True)
    i2.markdown(f'<div class="insight"><b>Role matters.</b><br>{role.index[0]}s have the highest attrition '
                f'({role.iloc[0] * 100:.1f}%), while {role.index[-1]}s have the lowest '
                f'({role.iloc[-1] * 100:.1f}%).</div>', unsafe_allow_html=True)
    i3.markdown(f'<div class="insight"><b>Accuracy can mislead.</b><br>Only {base * 100:.1f}% leave, so a model '
                f'that predicts "nobody leaves" is {100 - base * 100:.0f}% accurate but useless. We use recall '
                f'and F1.</div>', unsafe_allow_html=True)
    with st.expander("Preview the dataset"):
        st.dataframe(df.head(10))
    st.caption("Results show association between features and attrition, not cause.")

elif page == "explore":
    hero("Explore Data", "See how attrition changes across groups and features.")
    tab1, tab2, tab3 = st.tabs(["By category", "By number", "Correlations"])

    with tab1:
        cat_cols = [c for c in df.select_dtypes(exclude="number").columns if c != "Attrition" and c not in USELESS_COLS]
        col = st.selectbox("Attrition rate by:", cat_cols,
                           index=cat_cols.index("OverTime") if "OverTime" in cat_cols else 0)
        g = df.assign(flag=y_all).groupby(col)["flag"].agg(["mean", "size"]).sort_values("mean")
        g["rate"] = (g["mean"] * 100).round(1)
        g["color"] = ["Above average" if m > base else "Below average" for m in g["mean"]]
        fig = px.bar(g.reset_index(), x="rate", y=col, orientation="h", color="color",
                     color_discrete_map={"Above average": RED, "Below average": PRIMARY},
                     hover_data={"size": True, "color": False}, text="rate",
                     labels={"rate": "Attrition rate (%)", "size": "Employees"},
                     title=f"Attrition rate by {col}")
        fig.add_vline(x=base * 100, line_dash="dash", line_color="#F8FAFC",
                      annotation_text=f"Overall {base * 100:.1f}%")
        fig.update_layout(legend_title_text="")
        st.plotly_chart(style_fig(fig, 420))
        st.caption("Hover a bar to see how many employees are in the group. Small groups give less stable rates.")

    with tab2:
        num_cols = [c for c in df.select_dtypes(include="number").columns if c not in USELESS_COLS]
        ncol = st.selectbox("Feature:", num_cols,
                            index=num_cols.index("MonthlyIncome") if "MonthlyIncome" in num_cols else 0)
        fig = px.box(df, x="Attrition", y=ncol, color="Attrition",
                     color_discrete_map={"Yes": RED, "No": PRIMARY}, title=f"{ncol} for leavers vs stayers")
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 420))

    with tab3:
        corr = df.assign(AttritionFlag=y_all).select_dtypes(include="number").drop(columns=USELESS_COLS, errors="ignore").corr()
        c_att = corr["AttritionFlag"].drop("AttritionFlag").sort_values(key=abs, ascending=False).head(10).sort_values()
        fig = px.bar(x=c_att.values, y=c_att.index, orientation="h", color=c_att.values,
                     color_continuous_scale=[PRIMARY, "#334155", RED], range_color=[-0.4, 0.4],
                     labels={"x": "Correlation with attrition", "y": ""},
                     title="Features most correlated with attrition")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(fig, 400))
        with st.expander("Full correlation heatmap"):
            hm = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
            st.plotly_chart(style_fig(hm, 650))

elif page == "models":
    hero("Model Comparison", "Six models compared with 5-fold cross-validation, judged on recall and F1 for "
         "the 'Left' class.")
    with st.spinner("Running 5-fold cross-validation for 6 models (first load only)..."):
        results = run_cross_validation()
    best_f1 = results.sort_values("F1", ascending=False).iloc[0]
    best_rec = results.sort_values("Recall", ascending=False).iloc[0]
    c1, c2, c3 = st.columns(3)
    kpi(c1, "Best F1", best_f1["Model"], f"F1 = {best_f1['F1']:.3f}", "green")
    kpi(c2, "Best recall", best_rec["Model"], f"catches {best_rec['Recall'] * 100:.0f}% of leavers", "amber")
    kpi(c3, "Models compared", "6", "5-fold stratified cross-validation")

    st.write("")
    long = results.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1"], var_name="Metric", value_name="Score")
    fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group",
                 color_discrete_map={"Precision": SLATE, "Recall": RED, "F1": PRIMARY},
                 title="Precision, recall and F1 (Left class)")
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(style_fig(fig, 420))

    st.markdown('<div class="section-title">Full results</div>', unsafe_allow_html=True)
    st.dataframe(results.style.highlight_max(subset=["Precision", "Recall", "F1"], color="#4338CA")
                 .format({c: "{:.3f}" for c in ["Accuracy", "Precision", "Recall", "F1"]}))
    st.info("Why not accuracy? Predicting that nobody leaves already gives about 84% accuracy. A model can look "
            "accurate and still miss almost every employee who leaves (high precision but very low recall).")

elif page == "segments":
    hero("Employee Segments", "K-Means groups employees by similarity, without using the attrition column. "
         "Attrition is measured afterwards for each group.")
    seg, summary, profile = run_segmentation(5)
    risky = summary["AttritionRate"].idxmax()
    safest = summary["AttritionRate"].idxmin()
    share_leavers = (summary.loc[risky, "Size"] * summary.loc[risky, "AttritionRate"]) / y_all.sum()

    c1, c2, c3 = st.columns(3)
    kpi(c1, "Highest-risk segment", f"Cluster {risky}", f"{summary.loc[risky, 'AttritionRate'] * 100:.1f}% attrition", "red")
    kpi(c2, "Share of all leavers", f"{share_leavers * 100:.0f}%",
        f"from {summary.loc[risky, 'Size'] / len(df) * 100:.0f}% of employees", "amber")
    kpi(c3, "Lowest-risk segment", f"Cluster {safest}", f"{summary.loc[safest, 'AttritionRate'] * 100:.1f}% attrition", "green")

    st.write("")
    left, right = st.columns(2)
    with left:
        s = summary.reset_index()
        s["Cluster"] = s["Cluster"].astype(str)
        s["Rate %"] = (s["AttritionRate"] * 100).round(1)
        fig = px.bar(s, x="Cluster", y="Rate %", color="Rate %", color_continuous_scale=["#3730A3", RED],
                     hover_data={"Size": True}, title="Attrition rate by segment", text="Rate %")
        fig.add_hline(y=base * 100, line_dash="dash", line_color="#F8FAFC", annotation_text="Overall")
        fig.update_coloraxes(showscale=False)
        st.plotly_chart(style_fig(fig, 380))
    with right:
        plot = seg.copy()
        plot["Cluster"] = plot["Cluster"].astype(str)
        fig = px.scatter(plot, x="Age", y="MonthlyIncome", color="Cluster", opacity=0.7,
                         title="Segments: Age vs Monthly Income")
        st.plotly_chart(style_fig(fig, 380))

    st.markdown(f'<div class="section-title">Who is in the highest-risk segment (cluster {risky})?</div>',
                unsafe_allow_html=True)
    overall_mean, overall_std = seg[CLUSTER_COLS].mean(), seg[CLUSTER_COLS].std()
    diff = ((profile.loc[risky] - overall_mean) / overall_std).abs().sort_values(ascending=False).head(4)
    for feat in diff.index:
        if feat == "OverTime_Yes":
            st.write(f"- **{profile.loc[risky, feat] * 100:.0f}%** work overtime (overall {overall_mean[feat] * 100:.0f}%)")
        else:
            st.write(f"- **{feat}**: {profile.loc[risky, feat]:.1f} on average (overall {overall_mean[feat]:.1f})")
    with st.expander("Full segment profile (average values)"):
        st.dataframe(profile.T)
    st.caption("Segments overlap (low silhouette score), so treat them as useful groupings, not sharp boundaries.")

elif page == "predict":
    hero("Predict Attrition Risk", "Enter an employee's details to get a risk score from the Logistic Regression "
         "model. Other details use typical values.")
    model, scaler, columns = train_final_model()
    raw = df.drop(columns=["Attrition"] + [c for c in USELESS_COLS if c in df.columns])

    form, result = st.columns([1.1, 1])
    with form:
        with st.container(border=True):
            st.markdown('<div class="section-title">Employee details</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            overtime = a.selectbox("OverTime", sorted(raw["OverTime"].unique()), index=0)
            travel = b.selectbox("Business travel", sorted(raw["BusinessTravel"].unique()))
            marital = a.selectbox("Marital status", sorted(raw["MaritalStatus"].unique()))
            job_role = b.selectbox("Job role", sorted(raw["JobRole"].unique()))
            age = a.slider("Age", int(raw["Age"].min()), int(raw["Age"].max()), int(raw["Age"].median()))
            income = b.slider("Monthly income", int(raw["MonthlyIncome"].min()), int(raw["MonthlyIncome"].max()),
                              int(raw["MonthlyIncome"].median()))
            distance = a.slider("Distance from home", int(raw["DistanceFromHome"].min()),
                                int(raw["DistanceFromHome"].max()), int(raw["DistanceFromHome"].median()))
            years = b.slider("Years at company", 0, int(raw["YearsAtCompany"].max()), int(raw["YearsAtCompany"].median()))
            job_sat = a.slider("Job satisfaction (1-4)", 1, 4, 3)
            wlb = b.slider("Work-life balance (1-4)", 1, 4, 3)
            promo = a.slider("Years since last promotion", 0, int(raw["YearsSinceLastPromotion"].max()),
                             int(raw["YearsSinceLastPromotion"].median()))

    # start from a typical employee, then override with the inputs
    row = {c: (raw[c].median() if pd.api.types.is_numeric_dtype(raw[c]) else raw[c].mode()[0]) for c in raw.columns}
    row.update({"OverTime": overtime, "BusinessTravel": travel, "MaritalStatus": marital, "JobRole": job_role,
                "Age": age, "MonthlyIncome": income, "DistanceFromHome": distance, "JobSatisfaction": job_sat,
                "WorkLifeBalance": wlb, "YearsAtCompany": years, "YearsSinceLastPromotion": promo})
    combined = pd.concat([raw, pd.DataFrame([row])], ignore_index=True)
    x_new = pd.get_dummies(combined, drop_first=True).iloc[[-1]].reindex(columns=columns, fill_value=0).astype(float)
    x_scaled = scaler.transform(x_new)
    prob = model.predict_proba(x_scaled)[0, 1]
    level, color = ("Low", GREEN) if prob < 0.3 else (("Medium", AMBER) if prob < 0.6 else ("High", RED))

    with result:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#F8FAFC", "thickness": 0.25},
                   "steps": [{"range": [0, 30], "color": "rgba(52,211,153,0.40)"}, {"range": [30, 60], "color": "rgba(251,191,36,0.40)"},
                             {"range": [60, 100], "color": "rgba(248,113,113,0.45)"}]}))
        st.plotly_chart(style_fig(gauge, 270))
        st.markdown(f'<div style="text-align:center"><span class="badge" style="background:{color}">'
                    f'{level} risk</span></div>', unsafe_allow_html=True)
        st.caption("A model score, not an exact probability. The model gives extra weight to leavers, "
                   "so scores run higher than real-world rates.")

    contrib = pd.Series(model.coef_[0] * x_scaled[0], index=columns).sort_values()
    show = pd.concat([contrib.head(4), contrib.tail(4)])
    fig = px.bar(x=show.values, y=show.index, orientation="h", color=show.values > 0,
                 color_discrete_map={True: RED, False: GREEN},
                 labels={"x": "Pushes the score up (right) or down (left)", "y": ""},
                 title="What drives this employee's score")
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig, 360))
    st.caption("These show association with leaving in this dataset, not causes.")

elif page == "insights":
    hero("Insights & Recommendations", "What HR could investigate, based on the patterns found in the data.")
    flag = df.assign(flag=y_all)
    ot = flag.groupby("OverTime")["flag"].mean() * 100
    role_g = flag.groupby("JobRole")["flag"].agg(["mean", "size"]).sort_values("mean", ascending=False)
    trav = flag.groupby("BusinessTravel")["flag"].mean() * 100
    seg, summary, profile = run_segmentation(5)
    risky = summary["AttritionRate"].idxmax()
    share = summary.loc[risky, "Size"] * summary.loc[risky, "AttritionRate"] / y_all.sum() * 100
    top_role = role_g.index[0]

    a, b = st.columns(2)
    a.markdown(f'''<div class="insight"><b>1. Review overtime workload</b><br>{ot["Yes"]:.0f}% of employees working
        overtime left, against {ot["No"]:.0f}% of the rest.<br><br><i>Check whether overtime is concentrated in
        particular teams and whether it is voluntary.</i></div>''', unsafe_allow_html=True)
    b.markdown(f'''<div class="insight"><b>2. Look closely at {top_role}s</b><br>{role_g.iloc[0]["mean"] * 100:.0f}%
        of {top_role}s left ({int(role_g.iloc[0]["size"])} employees in the group).<br><br><i>Investigate pay,
        targets and career path for this role. The group is small, so confirm with more data.</i></div>''',
        unsafe_allow_html=True)
    st.write("")
    c, d = st.columns(2)
    c.markdown(f'''<div class="insight"><b>3. Focus on the highest-risk segment</b><br>Cluster {risky} has
        {summary.loc[risky, "AttritionRate"] * 100:.0f}% attrition and holds {share:.0f}% of all leavers;
        {profile.loc[risky, "OverTime_Yes"] * 100:.0f}% of it works overtime.<br><br><i>Start retention
        conversations with this group first.</i></div>''', unsafe_allow_html=True)
    d.markdown(f'''<div class="insight"><b>4. Review travel load</b><br>{trav["Travel_Frequently"]:.0f}% of
        frequent travelers left, against {trav["Non-Travel"]:.0f}% of non-travelers.<br><br><i>Check whether
        travel is shared fairly and whether it can be reduced.</i></div>''', unsafe_allow_html=True)

    st.write("")
    st.warning("**Use responsibly.** These are associations in a small, synthetic dataset, not proof of causes. "
               "Use the results to decide where to investigate, not to make decisions about individual "
               "employees. Do not use sensitive attributes such as gender or marital status to treat "
               "employees differently.")
    with st.expander("Limitations of this analysis"):
        st.write("- The test set has only 47 employees who left, so scores can shift with a different split.")
        st.write("- The dataset is synthetic and small, so results may not carry over to a real company.")
        st.write("- Segments overlap (low silhouette score), so they are useful groupings, not sharp boundaries.")
        st.write("- Model scores are not exact probabilities, because the models give extra weight to leavers.")

elif page == "quality":
    hero("Data Quality & Cleaning", "Real company data is rarely clean. This page finds missing values, "
         "duplicates, placeholder text and impossible negatives, cleans them, and logs every action.")
    mode = st.radio("Choose data", ["Messy-data demo (IBM data, deliberately corrupted)", "Upload my own CSV"],
                    horizontal=True)

    if mode.startswith("Messy"):
        messy = corrupt_data(df)
        s_before, t_before = quality_report(messy)
        cleaned, log = clean_data(messy)
        s_after, t_after = quality_report(cleaned)

        st.markdown('<div class="section-title">Before vs after cleaning</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for col, key in zip(cols, ["Rows", "Duplicate rows", "Missing values", "Negative values"]):
            col.metric(key, f"{s_after[key]:,}", f"{s_after[key] - s_before[key]:+,}",
                       delta_color="off" if key == "Rows" else "inverse")
        st.caption(f"Placeholder values (such as '?'): {s_before['Placeholder values']} before, "
                   f"{s_after['Placeholder values']} after.")

        left, right = st.columns([1.1, 1])
        with left:
            probs = t_before[(t_before.iloc[:, [0, 2, 3]] > 0).any(axis=1)][
                ["missing", "negative values", "placeholder text (?, NA...)"]]
            fig = px.bar(probs, barmode="stack", title="Problems found in the messy data, by column",
                         color_discrete_sequence=[AMBER, RED, PRIMARY], labels={"value": "Count", "index": ""})
            fig.update_layout(legend_title_text="")
            st.plotly_chart(style_fig(fig, 360))
        with right:
            st.markdown('<div class="section-title">Cleaning log</div>', unsafe_allow_html=True)
            for line in log:
                st.write("- " + line)

        st.markdown('<div class="section-title">Does cleaning recover model performance?</div>', unsafe_allow_html=True)
        f1_orig, f1_clean = lr_cv_f1(df), lr_cv_f1(cleaned)
        a, b = st.columns(2)
        kpi(a, "Original clean data", f"F1 = {f1_orig:.3f}", "Logistic Regression, 5-fold CV", "green")
        kpi(b, "Corrupted, then cleaned", f"F1 = {f1_clean:.3f}", f"{f1_clean - f1_orig:+.3f} vs original", "amber")
        st.caption("A small gap means the cleaning rebuilt data that is almost as useful as the original.")
    else:
        file = st.file_uploader("Upload a CSV file", type="csv")
        if file is not None:
            up = pd.read_csv(file)
            s_before, t_before = quality_report(up)
            cols = st.columns(3)
            for i, key in enumerate(["Rows", "Duplicate rows", "Missing values"]):
                kpi(cols[i], key, f"{s_before[key]:,}")
            st.dataframe(t_before)
            cleaned, log = clean_data(up)
            st.markdown('<div class="section-title">Cleaning log</div>', unsafe_allow_html=True)
            for line in log:
                st.write("- " + line)
            st.download_button("Download cleaned CSV", cleaned.to_csv(index=False).encode("utf-8"),
                               "cleaned_data.csv", "text/csv")
            st.caption("Negative values are treated as errors. Check that this is correct for your data.")

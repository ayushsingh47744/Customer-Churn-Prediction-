import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Customer Churn Analytics", layout="wide", page_icon="📊")

# ---------- Load data & model ----------
@st.cache_data
def load_data():
    df = pd.read_csv("customer_data.csv")
    return df

@st.cache_resource
def load_model():
    return joblib.load("best_churn_pipeline.pkl")

df = load_data()
model = load_model()

# Estimated average annual revenue per active customer (business assumption)
AVG_REVENUE_PER_CUSTOMER = 800

# ---------- Sidebar filters ----------
st.sidebar.title("🔎 Filters")
countries = st.sidebar.multiselect("Country", df["country"].unique(), default=list(df["country"].unique()))
genders = st.sidebar.multiselect("Gender", df["gender"].unique(), default=list(df["gender"].unique()))
active_filter = st.sidebar.selectbox("Membership Status", ["All", "Active Only", "Inactive Only"])

fdf = df[df["country"].isin(countries) & df["gender"].isin(genders)]
if active_filter == "Active Only":
    fdf = fdf[fdf["active_member"] == 1]
elif active_filter == "Inactive Only":
    fdf = fdf[fdf["active_member"] == 0]

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing **{len(fdf):,}** of {len(df):,} customers")

# ---------- Header ----------
st.title("📊 Customer Churn Analytics Dashboard")
st.caption("Business intelligence view for customer retention strategy")

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📈 Overview", "👥 Customer Segments", "⚠️ Risk Analysis", "💡 Retention Insights", "🔮 Predict Churn"]
)

# ===================== TAB 1: OVERVIEW =====================
with tab1:
    churned = int(fdf["churn"].sum())
    total = len(fdf)
    churn_rate = fdf["churn"].mean() * 100 if total else 0
    revenue_at_risk = churned * AVG_REVENUE_PER_CUSTOMER
    avg_balance = fdf["balance"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Churned Customers", f"{churned:,}")
    c3.metric("Churn Rate", f"{churn_rate:.1f}%")
    c4.metric("Est. Revenue at Risk", f"${revenue_at_risk:,.0f}")
    c5.metric("Avg. Balance", f"${avg_balance:,.0f}")

    st.divider()

    colA, colB = st.columns([2, 1])
    with colA:
        by_country = fdf.groupby("country").agg(
            customers=("customer_id", "count"),
            churn_rate=("churn", "mean")
        ).reset_index()
        by_country["churn_rate"] = by_country["churn_rate"] * 100
        fig = px.bar(by_country, x="country", y="churn_rate", color="country",
                     text=by_country["churn_rate"].round(1),
                     title="Churn Rate by Country (%)")
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        status = fdf["churn"].map({0: "Retained", 1: "Churned"}).value_counts().reset_index()
        status.columns = ["Status", "Count"]
        fig2 = px.pie(status, names="Status", values="Count", hole=0.5,
                      title="Retention Overview",
                      color="Status", color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"})
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Churn Rate by Age Group")
    fdf_age = fdf.copy()
    fdf_age["age_group"] = pd.cut(fdf_age["age"], bins=[0, 25, 35, 45, 55, 65, 100],
                                   labels=["<25", "25-34", "35-44", "45-54", "55-64", "65+"])
    age_churn = fdf_age.groupby("age_group", observed=True)["churn"].mean().reset_index()
    age_churn["churn"] = age_churn["churn"] * 100
    fig3 = px.line(age_churn, x="age_group", y="churn", markers=True,
                    title="Churn Rate Trend Across Age Groups")
    fig3.update_yaxes(title="Churn Rate (%)")
    st.plotly_chart(fig3, use_container_width=True)

# ===================== TAB 2: CUSTOMER SEGMENTS =====================
with tab2:
    st.subheader("Customer Segmentation Analysis")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(fdf, x="age", color="churn", barmode="overlay", nbins=30,
                            title="Age Distribution by Churn Status",
                            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"})
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        prod_churn = fdf.groupby("products_number")["churn"].mean().reset_index()
        prod_churn["churn"] = prod_churn["churn"] * 100
        fig2 = px.bar(prod_churn, x="products_number", y="churn",
                      title="Churn Rate by Number of Products", text=prod_churn["churn"].round(1))
        fig2.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.box(fdf, x="churn", y="estimated_salary", color="churn",
                      title="Estimated Salary vs Churn",
                      color_discrete_map={0: "#2ecc71", 1: "#e74c3c"})
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        gender_churn = fdf.groupby("gender")["churn"].mean().reset_index()
        gender_churn["churn"] = gender_churn["churn"] * 100
        fig4 = px.bar(gender_churn, x="gender", y="churn", color="gender",
                      title="Churn Rate by Gender", text=gender_churn["churn"].round(1))
        fig4.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("Segment Data Table")
    st.dataframe(
        fdf.groupby(["country", "gender"]).agg(
            customers=("customer_id", "count"),
            churn_rate=("churn", "mean"),
            avg_balance=("balance", "mean"),
            avg_salary=("estimated_salary", "mean")
        ).round(2).reset_index(),
        use_container_width=True
    )

# ===================== TAB 3: RISK ANALYSIS =====================
with tab3:
    st.subheader("What Drives Churn?")

    c1, c2 = st.columns(2)
    with c1:
        active_churn = fdf.groupby("active_member")["churn"].mean().reset_index()
        active_churn["active_member"] = active_churn["active_member"].map({0: "Inactive", 1: "Active"})
        active_churn["churn"] = active_churn["churn"] * 100
        fig = px.bar(active_churn, x="active_member", y="churn", color="active_member",
                     title="Churn Rate: Active vs Inactive Members",
                     text=active_churn["churn"].round(1),
                     color_discrete_map={"Active": "#2ecc71", "Inactive": "#e74c3c"})
        fig.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        cc_churn = fdf.groupby("credit_card")["churn"].mean().reset_index()
        cc_churn["credit_card"] = cc_churn["credit_card"].map({0: "No Card", 1: "Has Card"})
        cc_churn["churn"] = cc_churn["churn"] * 100
        fig2 = px.bar(cc_churn, x="credit_card", y="churn", color="credit_card",
                      title="Churn Rate: Credit Card Holders vs Non-Holders",
                      text=cc_churn["churn"].round(1))
        fig2.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()

    c3, c4 = st.columns(2)
    with c3:
        fig3 = px.box(fdf, x="churn", y="balance", color="churn",
                      title="Balance Distribution: Churned vs Retained",
                      color_discrete_map={0: "#2ecc71", 1: "#e74c3c"})
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fig4 = px.box(fdf, x="churn", y="tenure", color="churn",
                      title="Tenure Distribution: Churned vs Retained",
                      color_discrete_map={0: "#2ecc71", 1: "#e74c3c"})
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("Correlation Heatmap (Numeric Features)")
    num_cols = ["credit_score", "age", "tenure", "balance", "products_number",
                "estimated_salary", "churn"]
    corr = fdf[num_cols].corr()
    fig5 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                      title="Feature Correlation with Churn")
    st.plotly_chart(fig5, use_container_width=True)

# ===================== TAB 4: RETENTION INSIGHTS =====================
with tab4:
    st.subheader("💡 Key Business Insights & Recommendations")

    inactive_churn = df[df["active_member"] == 0]["churn"].mean() * 100
    active_churn_rate = df[df["active_member"] == 1]["churn"].mean() * 100
    high_bal_churn = df[df["balance"] > df["balance"].quantile(0.75)]["churn"].mean() * 100
    low_bal_churn = df[df["balance"] <= df["balance"].quantile(0.75)]["churn"].mean() * 100
    single_product_churn = df[df["products_number"] == 1]["churn"].mean() * 100
    multi_product_churn = df[df["products_number"] > 1]["churn"].mean() * 100

    insights = [
        {
            "title": "🔴 Inactive Members Churn Far More",
            "detail": f"Inactive members churn at **{inactive_churn:.1f}%** vs **{active_churn_rate:.1f}%** for active members.",
            "action": "Launch re-engagement campaigns (targeted offers, reminders) for inactive members before they leave."
        },
        {
            "title": "🟠 High-Balance Customers Are a Retention Priority",
            "detail": f"Customers with balances in the top 25% churn at **{high_bal_churn:.1f}%**, vs **{low_bal_churn:.1f}%** for others.",
            "action": "Assign relationship managers or loyalty perks to high-value account holders — losing them has outsized revenue impact."
        },
        {
            "title": "🟡 Single-Product Customers Are Flight Risks",
            "detail": f"Customers with only 1 product churn at **{single_product_churn:.1f}%**, vs **{multi_product_churn:.1f}%** for multi-product customers.",
            "action": "Cross-sell additional products/services to increase stickiness and reduce churn probability."
        },
    ]

    for ins in insights:
        with st.container(border=True):
            st.markdown(f"#### {ins['title']}")
            st.write(ins["detail"])
            st.info(f"**Recommended Action:** {ins['action']}")

    st.divider()
    st.subheader("Estimated Financial Impact")
    total_churned = df["churn"].sum()
    total_revenue_at_risk = total_churned * AVG_REVENUE_PER_CUSTOMER
    st.metric("Total Customers Churned (Full Dataset)", f"{total_churned:,}")
    st.metric("Estimated Annual Revenue at Risk", f"${total_revenue_at_risk:,.0f}",
               help=f"Assumes ${AVG_REVENUE_PER_CUSTOMER}/year average revenue per customer")

# ===================== TAB 5: PREDICT =====================
with tab5:
    st.subheader("🔮 Predict Churn for a New or Existing Customer")

    with st.form("prediction_form"):
        pc1, pc2, pc3 = st.columns(3)
        with pc1:
            credit_score = st.number_input("Credit Score", 300, 900, 650)
            country = st.selectbox("Country", df["country"].unique())
            gender = st.selectbox("Gender", df["gender"].unique())
            age = st.number_input("Age", 18, 100, 35)
        with pc2:
            tenure = st.number_input("Tenure (years)", 0, 15, 5)
            balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
            products_number = st.number_input("Number of Products", 1, 4, 1)
        with pc3:
            credit_card = st.selectbox("Has Credit Card", [1, 0])
            active_member = st.selectbox("Is Active Member", [1, 0])
            estimated_salary = st.number_input("Estimated Salary", 0.0, 300000.0, 60000.0)

        submitted = st.form_submit_button("Predict", type="primary")

        if submitted:
            input_df = pd.DataFrame([{
                "credit_score": credit_score,
                "country": country,
                "gender": gender,
                "age": age,
                "tenure": tenure,
                "balance": balance,
                "products_number": products_number,
                "credit_card": credit_card,
                "active_member": active_member,
                "estimated_salary": estimated_salary
            }])

            prediction = model.predict(input_df)[0]
            prob = model.predict_proba(input_df)[0][1]

            st.divider()
            colr1, colr2 = st.columns([1, 2])
            with colr1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={'text': "Churn Probability (%)"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#e74c3c" if prediction == 1 else "#2ecc71"},
                           'steps': [
                               {'range': [0, 33], 'color': "#d4f5dd"},
                               {'range': [33, 66], 'color': "#fdf3cf"},
                               {'range': [66, 100], 'color': "#fadbd8"}]}
                ))
                st.plotly_chart(fig, use_container_width=True)

            with colr2:
                if prediction == 1:
                    st.error(f"⚠️ This customer is **likely to churn** (probability: {prob*100:.1f}%).")
                    st.markdown("**Suggested retention actions:**")
                    if active_member == 0:
                        st.write("- Re-engagement campaign (inactive member)")
                    if products_number == 1:
                        st.write("- Offer additional product bundle")
                    if balance > df["balance"].quantile(0.75):
                        st.write("- Assign a relationship manager (high-value customer)")
                else:
                    st.success(f"✅ This customer is **likely to stay** (churn probability: {prob*100:.1f}%).")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 10px;'>
        Customer Retention Analytics © 2026 | Built with ❤️ using Streamlit & Scikit-Learn
        <br>
        Designed for Business Analytics
    </div>
    """,
    unsafe_allow_html=True
)
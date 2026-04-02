# ============================================
# TOURISM EXPERIENCE ANALYTICS - STREAMLIT APP
# ============================================
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)

# ============================================
# LOAD ALL MODELS & DATA
# ============================================
@st.cache_resource
def load_models():
    with open('models/regression_model.pkl', 'rb') as f:
        reg_model = pickle.load(f)
    with open('models/classification_model.pkl', 'rb') as f:
        cls_model = pickle.load(f)
    with open('models/content_similarity.pkl', 'rb') as f:
        content_sim = pickle.load(f)
    with open('models/attraction_features.pkl', 'rb') as f:
        attraction_features = pickle.load(f)
    with open('models/user_similarity.pkl', 'rb') as f:
        user_sim_df = pickle.load(f)
    with open('models/user_item_matrix.pkl', 'rb') as f:
        user_item_matrix = pickle.load(f)
    with open('models/reg_features.pkl', 'rb') as f:
        reg_features = pickle.load(f)
    with open('models/cls_features.pkl', 'rb') as f:
        cls_features = pickle.load(f)
    with open('models/top_users.pkl', 'rb') as f:
        top_users = pickle.load(f)
    return (reg_model, cls_model, content_sim, attraction_features,
            user_sim_df, user_item_matrix, reg_features,
            cls_features, top_users)

@st.cache_data
def load_data():
    return pd.read_csv('data/master_dataset.csv')

master = load_data()
(reg_model, cls_model, content_sim, attraction_features,
 user_sim_df, user_item_matrix, reg_features,
 cls_features, top_users) = load_models()

# ============================================
# ENCODERS (fit on master data)
# ============================================
le_continent    = LabelEncoder().fit(master['Continent'])
le_region       = LabelEncoder().fit(master['Region'])
le_country      = LabelEncoder().fit(master['Country'])
le_city         = LabelEncoder().fit(master['CityName'])
le_attrtype     = LabelEncoder().fit(master['AttractionType'])

visit_mode_map  = {1:'Business', 2:'Couples',
                   3:'Family',   4:'Friends', 5:'Solo'}

# ============================================
# SIDEBAR NAVIGATION
# ============================================
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/201/201623.png", width=80)
st.sidebar.title("🌍 Tourism Analytics")
page = st.sidebar.radio("Navigate", [
    "🏠 Home & EDA",
    "⭐ Predict Rating",
    "🧳 Predict Visit Mode",
    "🎯 Get Recommendations"
])

# ============================================
# PAGE 1: HOME & EDA
# ============================================
if page == "🏠 Home & EDA":
    st.title("🌍 Tourism Experience Analytics")
    st.markdown("### Explore Tourism Trends, Predict Ratings & Get Recommendations")

    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Visits",       f"{len(master):,}")
    col2.metric("Unique Users",        f"{master['UserId'].nunique():,}")
    col3.metric("Unique Attractions",  f"{master['AttractionId'].nunique():,}")
    col4.metric("Avg Rating",          f"{master['Rating'].mean():.2f}")

    st.markdown("---")

    # Plot 1: Rating Distribution
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Rating Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=master, x='Rating', palette='Blues', ax=ax)
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height()):,}',
                        (p.get_x() + p.get_width()/2, p.get_height()),
                        ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)

    # Plot 2: Visit Mode
    with col2:
        st.subheader("👫 Visit Mode Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        vc = master['VisitModeName'].value_counts()
        sns.barplot(x=vc.index, y=vc.values, palette='Set2', ax=ax)
        ax.set_xlabel("Visit Mode")
        ax.set_ylabel("Count")
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height()):,}',
                        (p.get_x() + p.get_width()/2, p.get_height()),
                        ha='center', va='bottom', fontsize=9)
        st.pyplot(fig)

    # Plot 3: Top Countries
    st.subheader("🌍 Top 10 Countries by Visits")
    fig, ax = plt.subplots(figsize=(12, 4))
    tc = master['Country'].value_counts().head(10)
    sns.barplot(x=tc.values, y=tc.index, palette='coolwarm', ax=ax)
    ax.set_xlabel("Number of Visits")
    st.pyplot(fig)

    # Plot 4 & 5
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏖️ Top Attraction Types")
        fig, ax = plt.subplots(figsize=(6, 4))
        tt = master['AttractionType'].value_counts().head(8)
        sns.barplot(x=tt.values, y=tt.index, palette='viridis', ax=ax)
        ax.set_xlabel("Visits")
        st.pyplot(fig)

    with col2:
        st.subheader("📅 Visits by Month")
        fig, ax = plt.subplots(figsize=(6, 4))
        month_names = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec']
        mc = master['VisitMonth'].value_counts().sort_index()
        sns.lineplot(x=month_names, y=mc.values,
                     marker='o', color='coral', ax=ax)
        ax.set_xlabel("Month")
        ax.set_ylabel("Visits")
        st.pyplot(fig)

    # Plot 6: Continent Pie
    st.subheader("🌐 Visits by Continent")
    fig, ax = plt.subplots(figsize=(6, 4))
    cc = master['Continent'].value_counts()
    ax.pie(cc.values, labels=cc.index, autopct='%1.1f%%',
           startangle=140,
           colors=sns.color_palette('Set3', len(cc)))
    st.pyplot(fig)

# ============================================
# PAGE 2: PREDICT RATING
# ============================================
elif page == "⭐ Predict Rating":
    st.title("⭐ Predict Attraction Rating")
    st.markdown("Fill in the details below to predict the rating.")

    col1, col2 = st.columns(2)
    with col1:
        visit_year   = st.selectbox("Visit Year",  [2020,2021,2022,2023])
        visit_month  = st.selectbox("Visit Month",
                                    list(range(1,13)),
                                    format_func=lambda x:
                                    ['Jan','Feb','Mar','Apr','May','Jun',
                                     'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
        continent    = st.selectbox("Continent",
                                    sorted(master['Continent'].unique()))
        region       = st.selectbox("Region",
                                    sorted(master['Region'].unique()))

    with col2:
        country      = st.selectbox("Country",
                                    sorted(master['Country'].unique()))
        city         = st.selectbox("City",
                                    sorted(master['CityName'].unique()))
        attraction   = st.selectbox("Attraction",
                                    sorted(master['Attraction'].unique()))
        attr_type    = st.selectbox("Attraction Type",
                                    sorted(master['AttractionType'].unique()))

    if st.button("🔮 Predict Rating", use_container_width=True):
        try:
            # Get IDs
            cont_id  = int(master[master['Continent']==continent]
                          ['ContinentId'].mode()[0])
            reg_id   = int(master[master['Region']==region]
                          ['RegionId'].mode()[0])
            cntry_id = int(master[master['Country']==country]
                          ['CountryId'].mode()[0])
            city_id  = int(master[master['CityName']==city]
                          ['CityId'].mode()[0])
            attr_id  = int(master[master['Attraction']==attraction]
                          ['AttractionId'].mode()[0])

            # Encoded values
            cont_enc = int(le_continent.transform([continent])[0])
            reg_enc  = int(le_region.transform([region])[0])
            cntry_enc= int(le_country.transform([country])[0])
            city_enc = int(le_city.transform([city])[0])
            atype_enc= int(le_attrtype.transform([attr_type])[0])

            # Aggregated features
            attr_avg = master[master['AttractionId']==attr_id]['Rating'].mean()
            attr_cnt = master[master['AttractionId']==attr_id].shape[0]
            user_cnt = 2  # default

            input_data = pd.DataFrame([{
                'VisitYear'         : visit_year,
                'VisitMonth'        : visit_month,
                'ContinentId'       : cont_id,
                'RegionId'          : reg_id,
                'CountryId'         : cntry_id,
                'CityId'            : city_id,
                'AttractionId'      : attr_id,
                'Continent_enc'     : cont_enc,
                'Region_enc'        : reg_enc,
                'Country_enc'       : cntry_enc,
                'CityName_enc'      : city_enc,
                'AttractionType_enc': atype_enc,
                'attr_avg_rating'   : attr_avg,
                'attr_visit_count'  : attr_cnt,
                'user_visit_count'  : user_cnt
            }])

            pred = reg_model.predict(input_data)[0]
            pred = np.clip(pred, 1, 5)

            st.success(f"⭐ Predicted Rating: **{pred:.2f} / 5.00**")
            st.progress(pred / 5.0)

            # Show attraction stats
            st.markdown("### 📊 Attraction Statistics")
            c1, c2, c3 = st.columns(3)
            c1.metric("Avg Rating",   f"{attr_avg:.2f}")
            c2.metric("Total Visits", f"{attr_cnt:,}")
            c3.metric("Predicted",    f"{pred:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")

# ============================================
# PAGE 3: PREDICT VISIT MODE
# ============================================
elif page == "🧳 Predict Visit Mode":
    st.title("🧳 Predict Visit Mode")
    st.markdown("Predict whether a visit is Business, Family, Couples, Friends or Solo.")

    col1, col2 = st.columns(2)
    with col1:
        visit_year  = st.selectbox("Visit Year",  [2020,2021,2022,2023])
        visit_month = st.selectbox("Visit Month",
                                   list(range(1,13)),
                                   format_func=lambda x:
                                   ['Jan','Feb','Mar','Apr','May','Jun',
                                    'Jul','Aug','Sep','Oct','Nov','Dec'][x-1])
        continent   = st.selectbox("Continent",
                                   sorted(master['Continent'].unique()))
        region      = st.selectbox("Region",
                                   sorted(master['Region'].unique()))
        user_avg_r  = st.slider("Your Average Rating", 1.0, 5.0, 4.0, 0.1)

    with col2:
        country     = st.selectbox("Country",
                                   sorted(master['Country'].unique()))
        city        = st.selectbox("City",
                                   sorted(master['CityName'].unique()))
        attraction  = st.selectbox("Attraction",
                                   sorted(master['Attraction'].unique()))
        attr_type   = st.selectbox("Attraction Type",
                                   sorted(master['AttractionType'].unique()))

    if st.button("🔮 Predict Visit Mode", use_container_width=True):
        try:
            cont_id  = int(master[master['Continent']==continent]
                          ['ContinentId'].mode()[0])
            reg_id   = int(master[master['Region']==region]
                          ['RegionId'].mode()[0])
            cntry_id = int(master[master['Country']==country]
                          ['CountryId'].mode()[0])
            city_id  = int(master[master['CityName']==city]
                          ['CityId'].mode()[0])
            attr_id  = int(master[master['Attraction']==attraction]
                          ['AttractionId'].mode()[0])

            cont_enc  = int(le_continent.transform([continent])[0])
            reg_enc   = int(le_region.transform([region])[0])
            cntry_enc = int(le_country.transform([country])[0])
            city_enc  = int(le_city.transform([city])[0])
            atype_enc = int(le_attrtype.transform([attr_type])[0])

            attr_avg  = master[master['AttractionId']==attr_id]['Rating'].mean()
            attr_cnt  = master[master['AttractionId']==attr_id].shape[0]
            user_cnt  = 2

            input_data = pd.DataFrame([{
                'VisitYear'         : visit_year,
                'VisitMonth'        : visit_month,
                'ContinentId'       : cont_id,
                'RegionId'          : reg_id,
                'CountryId'         : cntry_id,
                'CityId'            : city_id,
                'AttractionId'      : attr_id,
                'Continent_enc'     : cont_enc,
                'Region_enc'        : reg_enc,
                'Country_enc'       : cntry_enc,
                'CityName_enc'      : city_enc,
                'AttractionType_enc': atype_enc,
                'user_avg_rating'   : user_avg_r,
                'attr_avg_rating'   : attr_avg,
                'attr_visit_count'  : attr_cnt,
                'user_visit_count'  : user_cnt
            }])

            pred_mode = cls_model.predict(input_data)[0]
            mode_name = visit_mode_map[pred_mode]
            proba     = cls_model.predict_proba(input_data)[0]

            mode_icons = {
                'Business':'💼','Couples':'💑',
                'Family':'👨‍👩‍👧','Friends':'👫','Solo':'🧍'
            }
            st.success(f"{mode_icons[mode_name]} Predicted Visit Mode: **{mode_name}**")

            # Show probabilities
            st.markdown("### 📊 Probability by Visit Mode")
            classes = cls_model.classes_
            prob_df = pd.DataFrame({
                'Visit Mode': [visit_mode_map[c] for c in classes],
                'Probability': proba
            }).sort_values('Probability', ascending=False)

            fig, ax = plt.subplots(figsize=(8, 3))
            sns.barplot(data=prob_df, x='Visit Mode',
                        y='Probability', palette='Set2', ax=ax)
            ax.set_ylim(0, 1)
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}',
                           (p.get_x()+p.get_width()/2, p.get_height()),
                           ha='center', va='bottom')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")

# ============================================
# PAGE 4: RECOMMENDATIONS
# ============================================
elif page == "🎯 Get Recommendations":
    st.title("🎯 Personalized Attraction Recommendations")

    tab1, tab2 = st.tabs(["🏖️ Similar Attractions", "👤 For You (Collaborative)"])

    # --- CONTENT BASED ---
    with tab1:
        st.subheader("Find attractions similar to one you liked")
        top_attr_list = attraction_features['Attraction'].tolist()
        selected_attr = st.selectbox("Select an Attraction", top_attr_list)

        if st.button("🔍 Find Similar", use_container_width=True):
            attr_id = attraction_features[
                attraction_features['Attraction']==selected_attr
            ]['AttractionId'].values[0]

            idx = attraction_features[
                attraction_features['AttractionId']==attr_id].index[0]
            local_idx = attraction_features.index.get_loc(idx)

            sim_scores = list(enumerate(content_sim[local_idx]))
            sim_scores = sorted(sim_scores,
                                key=lambda x: x[1], reverse=True)
            sim_scores = [s for s in sim_scores
                          if attraction_features.iloc[s[0]]['AttractionId']
                          != attr_id][:5]

            rec_indices = [s[0] for s in sim_scores]
            result = attraction_features.iloc[rec_indices][[
                'Attraction', 'AttractionType',
                'Country', 'attr_avg_rating'
            ]].copy()
            result.columns = ['Attraction', 'Type', 'Country', 'Avg Rating']
            result['Avg Rating'] = result['Avg Rating'].round(2)
            result = result.reset_index(drop=True)
            result.index += 1

            st.markdown("### 🏆 Top 5 Similar Attractions")
            st.dataframe(result, use_container_width=True)

    # --- COLLABORATIVE ---
    with tab2:
        st.subheader("Get recommendations based on similar users")
        selected_user = st.selectbox("Select User ID",
                                     sorted(top_users[:100]))

        if st.button("🔍 Get My Recommendations", use_container_width=True):
            if selected_user not in user_sim_df.index:
                st.warning("User not found in system.")
            else:
                similar_users = (user_sim_df[selected_user]
                                 .sort_values(ascending=False)
                                 .iloc[1:11].index.tolist())

                master_full = load_data()
                user_visited = set(
                    master_full[master_full['UserId']==selected_user]
                    ['AttractionId'].tolist())

                recs = []
                for sim_user in similar_users:
                    sim_data = master_full[master_full['UserId']==sim_user]
                    for _, row in sim_data.iterrows():
                        if row['AttractionId'] not in user_visited:
                            recs.append({
                                'AttractionId'  : row['AttractionId'],
                                'Attraction'    : row['Attraction'],
                                'Type'          : row['AttractionType'],
                                'Country'       : row['Country'],
                                'Rating'        : row['Rating']
                            })

                if recs:
                    recs_df = (pd.DataFrame(recs)
                                 .groupby(['AttractionId','Attraction',
                                           'Type','Country'])['Rating']
                                 .mean().reset_index()
                                 .sort_values('Rating', ascending=False)
                                 .head(5))
                    recs_df['Rating'] = recs_df['Rating'].round(2)
                    recs_df = recs_df.drop('AttractionId', axis=1)
                    recs_df = recs_df.reset_index(drop=True)
                    recs_df.index += 1

                    st.markdown("### 🏆 Top 5 Recommended Attractions")
                    st.dataframe(recs_df, use_container_width=True)
                else:
                    st.info("No new recommendations found for this user.")
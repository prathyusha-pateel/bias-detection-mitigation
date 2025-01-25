import streamlit as st

def suggest_fairness_notions(task):

    fairness_notions = []

    # Step 2: Stakeholder Concerns
    if task:
        if task == "Classification":
            stakes = st.selectbox(
                "Is your task high stakes (e.g., life-altering, legal, or ethical impacts)?",
                ["Please Select", "High Stakes", "Low Stakes"]
            )
        elif task == "Regression":
            st.write("For regression tasks, fairness generally revolves around balancing errors across groups.")
        elif task == "Clustering":
            st.write("For clustering tasks, fairness often involves proportional representation and fair group formation.")

    # Step 4: Desired Fairness Outcomes
    if task == "Classification" and stakes:
        if stakes == "High Stakes":
            st.write("High stakes tasks require careful attention to fairness due to their potential societal impact. Select the fairness concerns most relevant to your task.")
            st.write("Select the most relevant fairness concerns:")
            t1 = st.checkbox("Representation in outcomes (Demographic Parity)  \n Demographic Parity: Ensures equal proportions of positive outcomes (e.g., loan approvals) across all groups. Example: Ensuring equal hiring rates for men and women.", value=False)
            t2 = st.checkbox("Equal error rates across groups (Equalized Odds)  \n Equalized Odds: Ensures equal false positive and false negative rates across groups. Example: Reducing false arrests in predictive policing for different racial groups.", value=False)
            t3 = st.checkbox("Avoiding missed opportunities (Equal Opportunity)  \n Equal Opportunity: Ensures equal true positive rates across groups. Example: Ensuring that qualified candidates from all demographics are hired equally.", value=False)
            if t1:
                fairness_notions.append("Demographic Parity")
            if t2:
                fairness_notions.append("Equalized Odds")
            if t3:
                fairness_notions.append("Equal Opportunity")
        elif stakes == "Low Stakes":
            st.write("Low stakes tasks often focus on improving prediction reliability while maintaining fairness.")
            st.write("Select the most relevant fairness concerns:")
            t1 = st.checkbox("Trustworthy positive predictions (Predictive Parity)  \n Predictive Parity: Ensures that the model's predictions are equally reliable across groups. Example: Ensuring that a credit score of 700 has the same likelihood of loan repayment for all demographics.", value=False)
            t2 = st.checkbox("Treating similar individuals alike (Individual Fairness)  \n Individual Fairness: Ensures that similar individuals receive similar outcomes. Example: Ensuring that two candidates with nearly identical resumes are treated equally", value=False)
            t3 = st.checkbox("Calibration  \n Calibration: Ensures that predicted probabilities are well-aligned with observed outcomes. Example: A 70% likelihood of an event corresponds to it actually occurring 70% of the time across all groups.", value=False)
            if t1:
                fairness_notions.append("Predictive Parity")
            if t2:
                fairness_notions.append("Individual Fairness")
            if t3:
                fairness_notions.append("Calibration")
    elif task == "Regression":
        st.header("Step 4: Desired Fairness Outcomes for Regression")
        st.write("Regression tasks involve predicting continuous values, and fairness focuses on error balance or individual treatment.")
        st.write("Select the most relevant fairness concerns:")
        t1 = st.checkbox("Group fairness (balancing prediction errors across groups)  \n Group Fairness for Regression: Ensures prediction errors are balanced across groups. Example: Ensuring salary predictions are equally accurate", value=False)
        t2 = st.checkbox("Individual fairness (ensuring similar predictions for similar individuals)  \n Individual Fairness for Regression: Ensures similar individuals receive similar predictions. Example: Predicting fair salaries for two candidates with nearly identical qualifications", value=False)
        t3 = st.checkbox("Mean Prediction Parity  \n Mean Prediction Parity: Ensures that the average predictions are the same across groups. Example: The average predicted salary", value=False)
        if t1:
            fairness_notions.append("Group Fairness for Regression")
        if t2:
            fairness_notions.append("Individual Fairness for Regression")
        if t3:
            fairness_notions.append("Mean Prediction Parity")
    elif task == "Clustering":
        st.header("Step 4: Desired Fairness Outcomes for Clustering")
        st.write("Clustering tasks focus on grouping data, and fairness often involves proportional representation or ensuring fair groupings.")
        st.write("Select the most relevant fairness concerns:")
        t1 = st.checkbox("Proportional representation in clusters  \n Proportional Representation: Ensures clusters reflect the diversity of the data. Example: In customer segmentation, ensuring minority groups are adequately represented.", value=False)
        t2 = st.checkbox("Fair distance metrics for grouping  \n Fair Distance Metrics: Adjusts distance calculations to ensure fairness in group formation. Example: Using fairness-aware metrics to avoid biased clustering based on sensitive attributes.", value=False)
        if t1:
            fairness_notions.append("Proportional Representation")
        if t2:
            fairness_notions.append("Fair Distance Metrics")
    
    return fairness_notions


    

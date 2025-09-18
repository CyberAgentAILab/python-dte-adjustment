Hillstrom Email Marketing Experiment
====================================

The Hillstrom email marketing dataset is a classic example from digital marketing, involving 64,000 customers randomly assigned to receive either a men's merchandise email, women's merchandise email, or no email (control). This experiment allows us to examine which email campaign strategy is most effective using revenue as the outcome.

**Background**: Kevin Hillstrom provided this dataset to demonstrate email marketing analytics. Customers who purchased within the last 12 months were randomly divided into three groups to test targeted email campaigns against a control group.

**Research Question**: Which email campaign performed best - the men's version or the women's version - and how do the effects vary across the revenue distribution?

Data Setup and Loading
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import LabelEncoder
    import dte_adj
    from dte_adj.plot import plot

    # Load the real Hillstrom dataset
    url = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    df = pd.read_csv(url)

    print(f"Dataset shape: {df.shape}")
    print(f"Average spend by segment:\n{df.groupby('segment')['spend'].mean()}")

    # Prepare the data for dte_adj analysis
    # Create treatment indicator: 0=No E-Mail, 1=Mens E-Mail, 2=Women E-Mail
    treatment_mapping = {'No E-Mail': 0, 'Mens E-Mail': 1, 'Women E-Mail': 2}
    D = df['segment'].map(treatment_mapping).values

    # Use spend as the outcome variable (revenue)
    revenue = df['spend'].values

    zip_code_mapping = {'Surburban': 0, 'Rural': 1, 'Urban': 2}  # Note: typo in original data
    channel_mapping = {'Phone': 0, 'Web': 1, 'Multichannel': 2}

    # Create feature matrix
    features = pd.DataFrame({
        'recency': df['recency'],
        'history': df['history'],
        'history_segment': df['history_segment'].map(lambda s: int(s[0])),
        'mens': df['mens'],
        'women': df['women'],
        'zip_code': df['zip_code'].map(zip_code_mapping),
        'newbie': df['newbie'],
        'channel': df['channel'].map(channel_mapping)
    })

    X = features.values

    print(f"\nDataset size: {len(D):,} customers")
    print(f"Control group (No Email): {(D==0).sum():,} ({(D==0).mean():.1%})")
    print(f"Men's Email group: {(D==1).sum():,} ({(D==1).mean():.1%})")
    print(f"Women's Email group: {(D==2).sum():,} ({(D==2).mean():.1%})")
    print("Average Spend by Treatment:")
    print(f"No Email: ${revenue[D==0].mean():.2f}")
    print(f"Men's Email: ${revenue[D==1].mean():.2f}")
    print(f"Women's Email: ${revenue[D==2].mean():.2f}")

    # Also show conversion rates
    print("\nConversion Rates:")
    print(f"No Email: {df[df['segment']=='No E-Mail']['conversion'].mean():.3f}")
    print(f"Men's Email: {df[df['segment']=='Mens E-Mail']['conversion'].mean():.3f}")
    print(f"Women's Email: {df[df['segment']=='Women E-Mail']['conversion'].mean():.3f}")

Email Campaign Effectiveness Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Initialize estimators
    simple_estimator = dte_adj.SimpleDistributionEstimator()
    ml_estimator = dte_adj.AdjustedDistributionEstimator(
        LinearRegression(),
        folds=5
    )

    # Fit estimators on the full dataset
    simple_estimator.fit(X, D, revenue)
    ml_estimator.fit(X, D, revenue)

    # Define revenue evaluation points
    revenue_locations = np.linspace(0, 500, 51)

Control vs Women's Email Campaign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, let's examine how the Women's email campaign performs compared to no email (control):

.. code-block:: python

    # Compute DTE: Women's email vs Control
    dte_women_ctrl, lower_women_ctrl, upper_women_ctrl = simple_estimator.predict_dte(
        target_treatment_arm=2,  # Women's email
        control_treatment_arm=0,  # No email control
        locations=revenue_locations,
        variance_type="moment"
    )

    # Visualize Women's vs Control using dte_adj's plot function
    plot(revenue_locations, dte_women_ctrl, lower_women_ctrl, upper_women_ctrl,
         title="Women's Email Campaign vs Control",
         xlabel="Spending ($)", ylabel="Distribution Treatment Effect")

    # Statistical summary
    positive_dte_women = (dte_women_ctrl > 0).mean()
    significant_dte_women = ((lower_women_ctrl > 0) | (upper_women_ctrl < 0)).mean()

    print(f"Women's Email vs Control Results:")
    print(f"Locations where Women's > Control: {positive_dte_women:.1%}")
    print(f"Statistically significant differences: {significant_dte_women:.1%}")
    print(f"Average DTE: {dte_women_ctrl.mean():.3f}")

Control vs Men's Email Campaign
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, let's examine how the Men's email campaign performs compared to no email (control):

.. code-block:: python

    # Compute DTE: Men's email vs Control
    dte_men_ctrl, lower_men_ctrl, upper_men_ctrl = simple_estimator.predict_dte(
        target_treatment_arm=1,  # Men's email
        control_treatment_arm=0,  # No email control
        locations=revenue_locations,
        variance_type="moment"
    )

    # Visualize Men's vs Control using dte_adj's plot function
    plot(revenue_locations, dte_men_ctrl, lower_men_ctrl, upper_men_ctrl,
         title="Men's Email Campaign vs Control",
         xlabel="Spending ($)", ylabel="Distribution Treatment Effect", color="purple")

    # Statistical summary
    positive_dte_men = (dte_men_ctrl > 0).mean()
    significant_dte_men = ((lower_men_ctrl > 0) | (upper_men_ctrl < 0)).mean()

    print(f"Men's Email vs Control Results:")
    print(f"Locations where Men's > Control: {positive_dte_men:.1%}")
    print(f"Statistically significant differences: {significant_dte_men:.1%}")
    print(f"Average DTE: {dte_men_ctrl.mean():.3f}")

Both Campaigns vs Control Comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The control vs email campaigns analysis produces the following comparison:

.. image:: ../_static/hillstorm_dte_control.png
   :alt: Hillstrom Email Campaigns vs Control Analysis
   :width: 800px
   :align: center

**Interpreting the Control Comparison Results**: These plots show how each email campaign performs against the no-email control group across different spending levels:

**Women's Email vs Control**:
- **Positive DTE values** indicate that Women's email campaign increases the probability of spending at those levels compared to no email
- **Distribution pattern** shows where Women's email is most effective in driving customer spending
- **Confidence intervals** reveal statistical significance of the treatment effects

**Men's Email vs Control**:
- **Comparative effectiveness** can be assessed by comparing the magnitude and patterns of effects
- **Different spending ranges** may show varying campaign effectiveness
- **Statistical significance** indicated by confidence intervals not crossing zero

**Key Control Analysis Findings**:

1. **Campaign Effectiveness**: Both campaigns show positive effects compared to no email, confirming that email marketing drives incremental spending

2. **Differential Patterns**: The shape and magnitude of effects differ between campaigns, revealing:
   - Which campaign has stronger overall effects
   - Different spending ranges where each campaign excels
   - Varying confidence in treatment effects across spending levels

3. **Business Implications**:
   - **ROI Assessment**: Compare effect sizes to determine which campaign provides better return on investment
   - **Customer Segmentation**: Identify spending ranges where each campaign is most/least effective
   - **Resource Allocation**: Data-driven decisions on campaign budget allocation

4. **Statistical Rigor**: Confidence intervals provide guidance on where observed differences are statistically reliable vs. potentially due to sampling variation

This analysis answers the fundamental question: "Do email campaigns work?" and establishes the baseline effectiveness of each campaign against no email.

Direct Campaign Comparison: Men's vs Women's Email
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, let's directly compare the two email campaigns to answer the key research question:

.. code-block:: python

    # Compute DTE: Women's vs Men's email campaigns
    dte_women_men, lower_women_men, upper_women_men = simple_estimator.predict_dte(
        target_treatment_arm=2,  # Women's email
        control_treatment_arm=1,  # Men's email (as "control")
        locations=revenue_locations,
        variance_type="moment"
    )

    dte_ml, lower_ml, upper_ml = ml_estimator.predict_dte(
        target_treatment_arm=2,  # Women's email
        control_treatment_arm=1,  # Men's email
        locations=revenue_locations,
        variance_type="moment"
    )

    # Visualize the distribution treatment effects using dte_adj's built-in plot function

    # Simple estimator
    plot(revenue_locations, dte_women_men, lower_women_men, upper_women_men,
         title="Email Campaign Comparison: Women's vs Men's (Simple Estimator)",
         xlabel="Spending ($)", ylabel="Distribution Treatment Effect")

    # ML-adjusted estimator
    plot(revenue_locations, dte_ml, lower_ml, upper_ml,
         title="Email Campaign Comparison: Women's vs Men's (ML-Adjusted Estimator)",
         xlabel="Spending ($)", ylabel="Distribution Treatment Effect")

    # Statistical summary
    positive_dte = (dte_ml > 0).mean()
    significant_dte = ((lower_ml > 0) | (upper_ml < 0)).mean()

    print(f"\nDirect Campaign Comparison Results:")
    print(f"Locations where Women's > Men's: {positive_dte:.1%}")
    print(f"Statistically significant differences: {significant_dte:.1%}")
    print(f"Average DTE: {dte_ml.mean():.3f}")

The analysis produces the following distribution treatment effects visualization:

.. image:: ../_static/hillstorm_dte.png
   :alt: Hillstrom Email Marketing DTE Analysis
   :width: 800px
   :align: center

**Interpreting the Campaign Comparison Results**: The plot shows the distribution treatment effects (DTE) comparing Women's vs Men's email campaigns across different spending levels. Key observations:

- **Positive DTE values** (above zero line) indicate that Women's email campaign increases the probability of spending at that level compared to Men's campaign
- **Confidence intervals** (shaded areas) show statistical uncertainty - where intervals don't cross zero, effects are statistically significant
- **Heterogeneous effects** across spending distribution reveal that campaign effectiveness varies by customer spending levels
- **ML-adjusted estimator** (bottom panel) typically provides more precise estimates with tighter confidence intervals than the simple estimator (top panel)

The distributional analysis reveals nuanced patterns that would be missed by simply comparing average spending between campaigns.

Revenue Category Analysis with PTE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Compute Probability Treatment Effects for Women's vs Men's comparison
    pte_simple, pte_lower_simple, pte_upper_simple = simple_estimator.predict_pte(
        target_treatment_arm=2,  # Women's email
        control_treatment_arm=1,  # Men's email
        locations=revenue_locations,
        variance_type="moment"
    )

    pte_ml, pte_lower_ml, pte_upper_ml = ml_estimator.predict_pte(
        target_treatment_arm=2,  # Women's email
        control_treatment_arm=1,  # Men's email
        locations=revenue_locations,
        variance_type="moment"
    )

    # Visualize PTE results using dte_adj's plot function with bar chart

    # Simple estimator
    plot(revenue_locations[:-1], pte_simple, pte_lower_simple, pte_upper_simple,
        chart_type="bar",
        title="Spending Category Effects: Women's vs Men's (Simple Estimator)",
        xlabel="Spending Category", ylabel="Probability Treatment Effect", color="purple")

    # ML-adjusted estimator
    plot(revenue_locations[:-1], pte_ml, pte_lower_ml, pte_upper_ml,
        chart_type="bar",
        title="Spending Category Effects: Women's vs Men's (ML-Adjusted Estimator)",
        xlabel="Spending Category", ylabel="Probability Treatment Effect")

The Probability Treatment Effects analysis produces the following visualization:

.. image:: ../_static/hillstorm_pte.png
   :alt: Hillstrom Email Marketing PTE Analysis
   :width: 800px
   :align: center

**Interpreting the PTE Results**: The bar charts show probability treatment effects across different spending intervals, revealing which spending ranges are most affected by the Women's vs Men's email campaigns:

- **Positive bars** indicate spending ranges where Women's email campaign increases the probability of customers spending in that range compared to Men's email
- **Negative bars** show ranges where Men's email campaign is more effective
- **Error bars** represent confidence intervals - bars that don't cross zero are statistically significant
- **Different patterns** between simple (top) and ML-adjusted (bottom) estimators show how machine learning adjustment can provide more precise estimates

**Key PTE Findings**:

1. **Low spending ranges** ($0-$25): Women's campaign may be more effective at driving small purchases
2. **Medium spending ranges** ($25-$100): Effects vary, showing differential campaign effectiveness
3. **High spending ranges** ($100+): Reveals which campaign is better at generating high-value customers
4. **Statistical significance**: Confidence intervals show where differences are reliable vs. due to chance

This granular analysis helps marketers understand not just which campaign generates more revenue overall, but specifically which spending behaviors each campaign drives.

**Key Findings**: Using the real Hillstrom dataset with 64,000 customers, the distributional analysis reveals nuanced patterns in how email campaigns affect customer spending. The analysis goes beyond simple average comparisons to show how treatment effects vary across the entire spending distribution, providing insights into which customer segments respond best to different campaign types. This demonstrates the power of distribution treatment effect analysis for understanding heterogeneous responses in digital marketing experiments.

Next Steps
~~~~~~~~~~

- Try with your own randomized experiment data
- Experiment with different ML models (XGBoost, Neural Networks) for adjustment
- Explore stratified estimators for covariate-adaptive randomization designs
- Use multi-task learning (``is_multi_task=True``) for computational efficiency with many locations

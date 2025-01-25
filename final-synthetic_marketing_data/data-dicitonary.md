# Synthetic Marketing Data Dictionary

## Core Data

### Demographics (demographics.csv)
- `SERIALNO`: Unique identifier
- `AGEP`: Age of person
- `SEX`: Gender
- `PINCP`: Personal income
- `ADJINC`: Income adjustment factor
- `PWGTP`: Person weight
- `STATE`: State FIPS code
- `PUMA`: Public Use Microdata Area
- `SCHL`: Educational attainment

### Consumer Preferences (consumer/[version]/consumer.csv)
- `consumer_id`: Unique consumer identifier
- `age_group`: Age bracket (18-34, 35-54, 55+)
- `online_shopping_rate`: Online purchase frequency
- `social_media_engagement_rate`: Social media activity level
- `loyalty_memberships`: Number of loyalty program memberships
- Various product preferences (e.g., ready_to_eat_preference, snacks_preference, etc.)
- Platform engagement metrics (e.g., facebook_engagement, instagram_engagement)
- Device usage patterns (mobile_usage, desktop_usage, etc.)

## Transaction Data

### Transactions (transactions/[version]/transactions.csv)
- `transaction_id`: Unique transaction identifier
- `consumer_id`: Link to consumer
- `transaction_date`: Date of purchase
- `channel`: Purchase channel (mobile, desktop, in_store)
- `transaction_value`: Total purchase amount
- `num_items`: Number of items in transaction
- `state`: State where transaction occurred
- `region`: Geographic region
- `buying_frequency`: Purchase frequency category

### Transaction Details (transactions/[version]/transaction_details.csv)
- `transaction_id`: Link to transaction
- `product_id`: Link to product
- `quantity`: Number of items
- `unit_price`: Price per unit
- `line_total`: Total line item amount
- `category`: Product category
- `subcategory`: Product subcategory

### Product Catalog (transactions/[version]/product_catalog.csv)
- `product_id`: Unique product identifier
- `category`: Product category
- `subcategory`: Product subcategory
- `base_price`: Standard price
- `min_order_quantity`: Minimum order amount
- `max_order_quantity`: Maximum order amount

## Marketing Data

### Campaigns (marketing_engagement/[version]/campaigns.csv)
- `campaign_id`: Unique campaign identifier
- `campaign_type`: Type of campaign
- `primary_channel`: Main marketing channel
- `creative_type`: Content format
- `creative_elements`: Content components
- `start_date`: Campaign start
- `end_date`: Campaign end
- `target_age_min/max`: Age targeting range
- `target_income_min/max`: Income targeting range
- `base_engagement_rate`: Expected engagement
- `budget`: Campaign budget
- `target_regions`: Geographic targeting

### Engagements (marketing_engagement/[version]/engagements.csv)
- `engagement_id`: Unique engagement identifier
- `campaign_id`: Link to campaign
- `date`: Engagement date
- `impressions`: Number of views
- `clicks`: Number of clicks
- `engagement_rate`: Interaction rate
- `time_spent_minutes`: Content viewing time
- `conversion_rate`: Conversion performance
- Campaign progress metrics

### Loyalties (marketing_engagement/[version]/loyalties.csv)
- `consumer_id`: Link to consumer
- `enrollment_date`: Program join date
- `status`: Membership status
- `points_balance`: Current points
- `lifetime_points`: Total points earned
- `tier`: Membership level
- `age_group`: Age bracket
- `redemption_rate`: Point usage rate
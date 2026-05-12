# Feature Engineering Documentation

## Overview

The BufferIQ feature engineering system transforms raw social media post data into ML-ready feature vectors for engagement prediction models. The system extracts 85+ features across five categories: temporal, content, NLP, engagement, and platform-specific.

## Architecture

### Design Principles

1. **Abstract Base Class Pattern**: All feature extractors inherit from `BaseFeatureExtractor`
2. **Batch + Single Extraction**: Support both DataFrame and dictionary inputs
3. **Platform Validation**: Strict validation (ONLY linkedin/twitter/bluesky)
4. **Graceful Degradation**: Missing columns trigger warnings, not errors
5. **Memory Efficiency**: Avoid unnecessary DataFrame copies
6. **Async Support**: Engagement features use AsyncSession for database queries

### Component Structure

```text
ml/features/
├── base.py                  # Abstract base class + platform validation
├── temporal.py              # Time-based features (21 features)
├── content.py               # Text/structure features (25 features)
├── nlp.py                   # Linguistic features (15 features)
├── engagement.py            # Historical features (15 features)
├── platform_specific.py     # Platform features (16 features)
├── scaler.py                # Feature scaling utilities
├── selector.py              # Feature selection utilities
└── pipeline.py              # Feature engineering orchestration
```
## Feature Categories

### 1. Temporal Features (21 features)

Extracted from `published_at` timestamp:

**Basic Time Features:**
- `hour` - Hour of day (0-23)
- `day_of_week` - Day of week (0=Monday, 6=Sunday)
- `day_of_month` - Day of month (1-31)
- `week_of_year` - ISO week number (1-52)
- `month` - Month (1-12)
- `quarter` - Quarter (1-4)
- `year` - Year

**Time of Day Indicators:**
- `is_weekend` - Boolean (Saturday/Sunday)
- `is_business_hours` - Boolean (9 AM - 5 PM)
- `is_morning` - Boolean (6 AM - 12 PM)
- `is_afternoon` - Boolean (12 PM - 5 PM)
- `is_evening` - Boolean (5 PM - 10 PM)
- `is_night` - Boolean (10 PM - 6 AM)
- `is_peak_hour` - Boolean (platform-specific optimal hours)

**Time Calculations:**
- `time_since_midnight` - Minutes since midnight
- `time_until_midnight` - Minutes until midnight

**Recency Features:**
- `days_since_last_post` - Days since user's last post
- `hours_since_last_post` - Hours since user's last post
- `posts_in_last_24h` - Count of posts in last 24 hours
- `posts_in_last_7d` - Count of posts in last 7 days
- `avg_posting_interval_hours` - Average hours between posts

**Platform Peak Hours:**
- LinkedIn: 8-10 AM, 12-1 PM, 5-6 PM (weekdays)
- Twitter: 12-1 PM, 5-6 PM (weekdays), 9-10 AM (weekends)
- Bluesky: Similar to Twitter

### 2. Content Features (25 features)

Extracted from post `content` text:

**Text Length:**
- `text_length` - Total character count
- `word_count` - Total word count
- `avg_word_length` - Average characters per word
- `sentence_count` - Number of sentences
- `avg_sentence_length` - Average words per sentence
- `paragraph_count` - Number of paragraphs

**Formatting:**
- `has_url` - Boolean (contains URL)
- `url_count` - Number of URLs
- `has_hashtag` - Boolean (contains #)
- `hashtag_count` - Number of hashtags
- `has_mention` - Boolean (contains @)
- `mention_count` - Number of mentions
- `has_emoji` - Boolean (contains emoji)
- `emoji_count` - Number of emojis
- `has_number` - Boolean (contains digits)
- `number_count` - Count of numeric tokens
- `has_question` - Boolean (contains ?)
- `question_count` - Number of question marks
- `has_exclamation` - Boolean (contains !)
- `exclamation_count` - Number of exclamation marks

**Punctuation:**
- `uppercase_ratio` - Proportion of uppercase letters
- `punctuation_ratio` - Proportion of punctuation
- `special_char_count` - Count of special characters
- `newline_count` - Number of line breaks
- `whitespace_ratio` - Proportion of whitespace

### 3. NLP Features (15 features)

Extracted using TextBlob and textstat:

**Sentiment (TextBlob):**
- `sentiment_polarity` - -1 (negative) to +1 (positive)
- `sentiment_subjectivity` - 0 (objective) to 1 (subjective)
- `sentiment_label` - Categorical (-1, 0, 1)

**Readability:**
- `flesch_reading_ease` - 0-100 (higher = easier)
- `flesch_kincaid_grade` - US grade level
- `automated_readability_index` - ARI score
- `coleman_liau_index` - CLI score
- `avg_readability` - Normalized average of all scores

**Linguistic:**
- `lexical_diversity` - Unique words / total words
- `stopword_ratio` - Proportion of stopwords
- `noun_count` - Count of nouns
- `verb_count` - Count of verbs
- `adjective_count` - Count of adjectives
- `adverb_count` - Count of adverbs
- `proper_noun_count` - Count of proper nouns

### 4. Engagement Features (15 features)

Extracted from historical engagement data:

**User Baselines:**
- `user_avg_likes` - User's average likes
- `user_avg_comments` - User's average comments
- `user_avg_shares` - User's average shares
- `user_avg_engagement_rate` - User's average engagement rate
- `user_median_engagement_rate` - User's median engagement rate
- `user_post_count` - Total posts by user

**Platform Baselines:**
- `platform_avg_likes` - Platform average likes
- `platform_avg_comments` - Platform average comments
- `platform_avg_shares` - Platform average shares
- `platform_avg_engagement_rate` - Platform average engagement rate

**Rolling Windows:**
- `engagement_rate_last_5` - Average of last 5 posts
- `engagement_rate_last_10` - Average of last 10 posts
- `engagement_trend` - Slope of last 10 posts
- `is_improving` - Boolean (engagement trending up)
- `best_post_engagement` - Best rate in last 30 days

### 5. Platform-Specific Features (16 features)

**LinkedIn Features:**
- `is_professional_tone` - Formal language detection
- `has_career_keywords` - Job/hiring/opportunity keywords
- `has_industry_hashtags` - Industry-specific tags
- `optimal_length_linkedin` - 1300-1500 characters
- `has_call_to_action` - Common CTAs
- `document_structure_score` - Paragraph/formatting quality

**Twitter Features:**
- `is_thread_starter` - Ends with 🧵 or "Thread:"
- `has_retweet_keywords` - RT, via, etc.
- `is_reply` - Starts with @
- `uses_twitter_lingo` - TIL, IMO, ICYMI, etc.
- `optimal_length_twitter` - 71-100 characters
- `hashtag_position` - 0=none, 1=first, 2=middle, 3=end

**Bluesky Features:**
- `is_decentralization_topic` - AT Protocol, federation, etc.
- `has_tech_keywords` - Tech-focused keywords
- `community_engagement_style` - Conversational vs broadcast
- `optimal_length_bluesky` - 71-100 characters

## Platform Support

**CRITICAL: Only 3 platforms are supported:**
- `linkedin` - LinkedIn posts
- `twitter` - Twitter/X posts
- `bluesky` - Bluesky Social posts

**Facebook is NOT supported.** All platform validation rejects non-supported platforms.

## Usage Examples

### Basic Feature Extraction

```python
import pandas as pd
from bufferiq.ml.features.pipeline import FeatureEngineeringPipeline

# Load your data
df = pd.DataFrame({
    "published_at": ["2024-01-01T10:00:00Z"],
    "content": "Hello world! #test",
    "platform": "linkedin"
})

# Create pipeline with all extractors
pipeline = FeatureEngineeringPipeline()

# Extract features
features = await pipeline.extract_features(df)

print(f"Extracted {len(features.columns)} features")
```

### Extract Specific Feature Sets

```python
from bufferiq.ml.features.temporal import TemporalFeatureExtractor
from bufferiq.ml.features.content import ContentFeatureExtractor
from bufferiq.ml.features.pipeline import FeatureEngineeringPipeline

# Use only temporal and content features
extractors = [
    TemporalFeatureExtractor(),
    ContentFeatureExtractor(),
]

pipeline = FeatureEngineeringPipeline(extractors=extractors)
features = await pipeline.extract_features(df)
```

### Feature Scaling

```python
from bufferiq.ml.features.scaler import FeatureScaler
from bufferiq.ml.features.pipeline import FeatureEngineeringPipeline

# Create scaler
scaler = FeatureScaler(method="standard")

# Create pipeline with scaler
pipeline = FeatureEngineeringPipeline(scaler=scaler)

# Extract and scale features
features = await pipeline.extract_features(
    df,
    fit_scaler=True  # Fit scaler on this data
)

# Save scaler for later use
scaler.save("outputs/features/scaler.joblib")
```

### Feature Selection

```python
from bufferiq.ml.features.selector import FeatureSelector
from bufferiq.ml.features.pipeline import FeatureEngineeringPipeline

# Create selector
selector = FeatureSelector(method="mutual_info", k=20)

# Create pipeline with selector
pipeline = FeatureEngineeringPipeline(selector=selector)

# Extract and select features
features = await pipeline.extract_features(
    df,
    fit_selector=True,
    target_column="engagement_rate"
)

# Get selected features
selected = selector.get_selected_features()
print(f"Selected {len(selected)} features")

# Get feature importance
importance = selector.get_feature_importance()
print(importance.head(10))
```

### Single Post Extraction

```python
from bufferiq.ml.features.temporal import TemporalFeatureExtractor

extractor = TemporalFeatureExtractor()

post_data = {
    "published_at": "2024-01-01T10:00:00Z",
    "platform": "linkedin"
}

features = extractor.extract_single(post_data)
print(features)
```

## CLI Commands

### Extract Features

```bash
# Extract all features for user
python -m bufferiq.cli.features extract --user-id=1 --output features.csv

# Extract specific feature sets
python -m bufferiq.cli.features extract \
    --user-id=1 \
    --extractors temporal,content,nlp \
    --output features.csv

# Extract with statistics
python -m bufferiq.cli.features extract --user-id=1 --stats

# Extract and fit scaler
python -m bufferiq.cli.features extract \
    --user-id=1 \
    --fit-scaler \
    --save-scaler scaler.joblib

# Extract and fit selector
python -m bufferiq.cli.features extract \
    --user-id=1 \
    --fit-selector \
    --save-selector selector.joblib \
    --target engagement_rate
```

### List Features

```bash
# Show all available features
python -m bufferiq.cli.features list-features
```

### Analyze Feature Importance

```bash
# Show top 20 most important features
python -m bufferiq.cli.features importance \
    --user-id=1 \
    --target engagement_rate \
    --top 20

# Save importance to CSV
python -m bufferiq.cli.features importance \
    --user-id=1 \
    --target engagement_rate \
    --top 20 \
    --output feature_importance.csv
```

## Script Usage

```bash
# Extract features for user
python scripts/extract_features.py --user-id=1

# Extract with platform filter
python scripts/extract_features.py --user-id=1 --platform linkedin

# Extract and save
python scripts/extract_features.py \
    --user-id=1 \
    --output outputs/features/features.csv \
    --save-scaler
```

## Scaling Methods

### StandardScaler (default)
- Mean = 0, Std = 1
- Assumes normal distribution
- Best for: Most ML algorithms (SVM, neural networks, linear regression)

```python
scaler = FeatureScaler(method="standard")
```

### MinMaxScaler
- Scales to [0, 1] range
- Preserves zero values
- Best for: Bounded features, neural networks

```python
scaler = FeatureScaler(method="minmax")
```

### RobustScaler
- Uses median and IQR
- Robust to outliers
- Best for: Data with outliers

```python
scaler = FeatureScaler(method="robust")
```

## Selection Methods

### Variance Threshold
- Removes low-variance features
- No target required

```python
selector = FeatureSelector(method="variance", threshold=0.0)
```

### Correlation Filter
- Removes highly correlated features
- No target required

```python
selector = FeatureSelector(method="correlation", threshold=0.95)
```

### Mutual Information
- Selects features based on mutual info with target
- Requires target

```python
selector = FeatureSelector(method="mutual_info", k=20)
```

### K-Best (F-regression)
- Selects top K features using F-scores
- Requires target

```python
selector = FeatureSelector(method="k_best", k=20)
```

## Performance

### Benchmarks (1000 posts)

- **Temporal Features**: < 1 second
- **Content Features**: < 2 seconds
- **NLP Features**: < 3 seconds
- **Engagement Features**: < 2 seconds
- **Platform Features**: < 1 second
- **Total Pipeline**: < 5 seconds
- **Memory Usage**: < 200MB

### Optimization Tips

1. **Use specific extractors**: Only load extractors you need
2. **Batch processing**: Process in batches of 1000 posts
3. **Cache results**: Save extracted features to disk
4. **Async operations**: Use async for engagement features
5. **Feature selection**: Reduce dimensionality early

## Error Handling

### Missing Columns

```python
# Graceful degradation - logs warning, continues
features = extractor.extract(df_missing_cols)
```

### Empty Content

```python
# Returns zero features
features = extractor.extract_single({"content": ""})
```

### Invalid Platform

```python
# Raises ValueError
try:
    validate_platform("facebook")
except ValueError as e:
    print(e)  # "Platform 'facebook' is not supported..."
```

## Testing

### Run All Tests

```bash
# Run all feature tests with coverage
pytest backend/tests/test_*_features.py \
    backend/tests/test_feature_*.py \
    -v --cov=bufferiq/ml/features \
    --cov-report=term-missing \
    --cov-fail-under=90
```

### Run Specific Test Suite

```bash
pytest backend/tests/test_temporal_features.py -v
pytest backend/tests/test_platform_features.py -v
```

## Integration with ML Pipeline

```python
# Day 7: Feature Engineering
pipeline = FeatureEngineeringPipeline()
features = await pipeline.extract_features(df, fit_scaler=True)

# Day 8: Model Training (next step)
from sklearn.ensemble import RandomForestRegressor

X = features
y = df["engagement_rate"]

model = RandomForestRegressor()
model.fit(X, y)
```

## Next Steps (Days 8-14)

1. **Day 8**: ML training pipeline
2. **Day 9-10**: Engagement prediction model
3. **Day 11**: Model evaluation
4. **Day 12-13**: Model optimization
5. **Day 14**: Model serving

## Troubleshooting

### TextBlob/NLTK Not Found

```bash
pip install textblob nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### textstat Not Found

```bash
pip install textstat
```

### Platform Validation Errors

Ensure platform is one of: `linkedin`, `twitter`, `bluesky`

### Memory Issues

Process in smaller batches:

```python
batch_size = 500
for i in range(0, len(df), batch_size):
    batch = df[i:i+batch_size]
    features = await pipeline.extract_features(batch)
    # Save batch
```

## API Reference

See individual module docstrings for detailed API documentation:

- `bufferiq.ml.features.base.BaseFeatureExtractor`
- `bufferiq.ml.features.temporal.TemporalFeatureExtractor`
- `bufferiq.ml.features.content.ContentFeatureExtractor`
- `bufferiq.ml.features.nlp.NLPFeatureExtractor`
- `bufferiq.ml.features.engagement.EngagementFeatureExtractor`
- `bufferiq.ml.features.platform_specific.PlatformSpecificFeatureExtractor`
- `bufferiq.ml.features.scaler.FeatureScaler`
- `bufferiq.ml.features.selector.FeatureSelector`
- `bufferiq.ml.features.pipeline.FeatureEngineeringPipeline`


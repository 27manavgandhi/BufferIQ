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

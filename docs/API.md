\# BufferIQ API Documentation



\## Overview



The BufferIQ API is a production-grade RESTful API for predicting social media engagement. It provides endpoints for single predictions, batch predictions, model management, and health monitoring.



\*\*Base URL:\*\* `http://localhost:8000` (development)  

\*\*Production URL:\*\* `https://api.bufferiq.com`



\*\*Version:\*\* 1.0.0



\## Table of Contents



\- \[Authentication](#authentication)

\- \[Rate Limiting](#rate-limiting)

\- \[Endpoints](#endpoints)

&#x20; - \[Predictions](#predictions)

&#x20; - \[Batch Predictions](#batch-predictions)

&#x20; - \[Models](#models)

&#x20; - \[Health](#health)

&#x20; - \[Metrics](#metrics)

\- \[Request/Response Format](#requestresponse-format)

\- \[Error Handling](#error-handling)

\- \[Examples](#examples)



\## Authentication



\*\*Current Status:\*\* No authentication required for development.



\*\*Production:\*\* API key authentication will be required:



```bash

curl -H "Authorization: Bearer YOUR\_API\_KEY" \\

&#x20; https://api.bufferiq.com/api/v1/predict

```



\## Rate Limiting



\*\*Development:\*\* 60 requests per minute  

\*\*Production:\*\* 1000 requests per minute



Rate limit headers included in responses:

\- `X-RateLimit-Limit`: Maximum requests allowed

\- `X-RateLimit-Remaining`: Requests remaining

\- `X-RateLimit-Reset`: Time when limit resets



\*\*429 Response when rate limited:\*\*

```json

{

&#x20; "detail": "Rate limit exceeded. Try again in 42 seconds."

}

```



\## Endpoints



\### Predictions



\#### POST /api/v1/predict



Predict engagement for a single social media post.



\*\*Request Body:\*\*

```json

{

&#x20; "content": "Just shipped a new feature! 🚀",

&#x20; "platform": "linkedin",

&#x20; "scheduled\_time": "2026-04-30T14:00:00Z",

&#x20; "has\_media": false,

&#x20; "has\_link": true

}

```



\*\*Parameters:\*\*

\- `content` (string, required): Post content (1-10000 characters)

\- `platform` (string, required): Platform - `linkedin`, `twitter`, or `bluesky`

\- `scheduled\_time` (datetime, optional): When post will be published

\- `post\_type` (string, optional): Type of post (default: "text")

\- `has\_media` (boolean, optional): Whether post includes media (default: false)

\- `has\_link` (boolean, optional): Whether post includes link (default: false)



\*\*Response (200 OK):\*\*

```json

{

&#x20; "engagement\_score": 7.8,

&#x20; "confidence": 0.85,

&#x20; "breakdown": {

&#x20;   "likes": 45,

&#x20;   "comments": 8,

&#x20;   "shares": 3

&#x20; },

&#x20; "metadata": {

&#x20;   "model\_version": "ensemble\_v1.0.0",

&#x20;   "inference\_time\_ms": 45.2,

&#x20;   "features\_used": 92,

&#x20;   "cached": false,

&#x20;   "timestamp": "2026-04-27T10:30:00Z"

&#x20; }

}

```



\*\*Query Parameters:\*\*

\- `model\_name` (optional): Specific model to use (default: "ensemble")



\*\*Example:\*\*

```bash

curl -X POST http://localhost:8000/api/v1/predict \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{

&#x20;   "content": "Just shipped a new feature!",

&#x20;   "platform": "linkedin"

&#x20; }'

```



\#### POST /api/v1/predict/ensemble



Explicitly use ensemble model for prediction (convenience endpoint).



Same request/response format as `/predict`.



\### Batch Predictions



\#### POST /api/v1/batch/predict



Predict engagement for multiple posts in a single request.



\*\*Request Body:\*\*

```json

{

&#x20; "items": \[

&#x20;   {

&#x20;     "id": "post\_1",

&#x20;     "request": {

&#x20;       "content": "First post",

&#x20;       "platform": "linkedin"

&#x20;     }

&#x20;   },

&#x20;   {

&#x20;     "id": "post\_2",

&#x20;     "request": {

&#x20;       "content": "Second post",

&#x20;       "platform": "twitter"

&#x20;     }

&#x20;   }

&#x20; ]

}

```



\*\*Parameters:\*\*

\- `items` (array, required): Array of prediction items (1-100 items)

&#x20; - `id` (string, required): Unique identifier for this item

&#x20; - `request` (object, required): Prediction request (same as single prediction)



\*\*Response (200 OK):\*\*

```json

{

&#x20; "predictions": \[

&#x20;   {

&#x20;     "id": "post\_1",

&#x20;     "prediction": {

&#x20;       "engagement\_score": 7.5,

&#x20;       "confidence": 0.82,

&#x20;       "breakdown": {...},

&#x20;       "metadata": {...}

&#x20;     }

&#x20;   },

&#x20;   {

&#x20;     "id": "post\_2",

&#x20;     "prediction": {

&#x20;       "engagement\_score": 6.2,

&#x20;       "confidence": 0.79,

&#x20;       "breakdown": {...},

&#x20;       "metadata": {...}

&#x20;     }

&#x20;   }

&#x20; ],

&#x20; "metadata": {

&#x20;   "total\_items": 2,

&#x20;   "processing\_time\_ms": 125.5,

&#x20;   "cache\_hits": 0,

&#x20;   "errors": 0

&#x20; }

}

```



\*\*Limits:\*\*

\- Maximum 100 items per batch

\- Each item subject to same validation as single prediction



\*\*Example:\*\*

```bash

curl -X POST http://localhost:8000/api/v1/batch/predict \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{

&#x20;   "items": \[

&#x20;     {

&#x20;       "id": "post\_1",

&#x20;       "request": {

&#x20;         "content": "First post",

&#x20;         "platform": "linkedin"

&#x20;       }

&#x20;     }

&#x20;   ]

&#x20; }'

```



\### Models



\#### GET /api/v1/models



List all available models.



\*\*Response (200 OK):\*\*

```json

{

&#x20; "models": \[

&#x20;   "xgboost",

&#x20;   "lightgbm",

&#x20;   "random\_forest",

&#x20;   "ensemble"

&#x20; ],

&#x20; "loaded": \[

&#x20;   "ensemble"

&#x20; ]

}

```



\#### GET /api/v1/models/{model\_name}



Get information about a specific model.



\*\*Response (200 OK):\*\*

```json

{

&#x20; "name": "ensemble",

&#x20; "path": "outputs/models/ensembles/production\_ensemble.joblib",

&#x20; "loaded": true,

&#x20; "exists": true

}

```



\#### POST /api/v1/models/{model\_name}/reload



Reload a specific model (admin only in production).



\*\*Response (200 OK):\*\*

```json

{

&#x20; "status": "success",

&#x20; "message": "Model ensemble reloaded"

}

```



\### Health



\#### GET /health



Overall health check.



\*\*Response (200 OK):\*\*

```json

{

&#x20; "status": "healthy",

&#x20; "services": {

&#x20;   "cache": {

&#x20;     "status": "healthy",

&#x20;     "message": "Cache is responding",

&#x20;     "response\_time\_ms": 2.5

&#x20;   },

&#x20;   "models": {

&#x20;     "status": "healthy",

&#x20;     "message": "3/4 models loaded"

&#x20;   }

&#x20; },

&#x20; "timestamp": "2026-04-27T10:30:00Z"

}

```



\#### GET /health/ready



Readiness probe for Kubernetes.



\*\*Response (200 OK):\*\*

```json

{

&#x20; "status": "ready",

&#x20; "message": "3 models loaded"

}

```



\*\*Response (503 Service Unavailable):\*\*

```json

{

&#x20; "detail": "Service not ready - no models loaded"

}

```



\#### GET /health/live



Liveness probe for Kubernetes.



\*\*Response (200 OK):\*\*

```json

{

&#x20; "status": "alive"

}

```



\### Metrics



\#### GET /metrics



Prometheus metrics endpoint.



\*\*Response (200 OK):\*\*

```text

HELP bufferiq\_requests\_total Total number of requests

TYPE bufferiq\_requests\_total counter

bufferiq\_requests\_total{endpoint="predict",platform="linkedin"} 150.0

HELP bufferiq\_request\_duration\_seconds Request duration in seconds

TYPE bufferiq\_request\_duration\_seconds histogram

bufferiq\_request\_duration\_seconds\_bucket{endpoint="predict",le="0.01"} 45.0

```



\## Request/Response Format



\### Content Type



All requests must use `application/json` content type.



\### Response Headers



All responses include:

\- `X-Request-ID`: Unique request identifier

\- `X-Process-Time`: Processing time in milliseconds

\- `Content-Type`: `application/json`



\### Timestamps



All timestamps are in ISO 8601 format with UTC timezone:



```text

2026-04-30T14:00:00Z

```

\## Error Handling



\### Error Response Format



```json

{

&#x20; "detail": "Error message",

&#x20; "errors": \[

&#x20;   {

&#x20;     "field": "platform",

&#x20;     "message": "Platform 'facebook' not supported",

&#x20;     "type": "value\_error"

&#x20;   }

&#x20; ]

}

```



\### HTTP Status Codes



\- \*\*200 OK\*\*: Successful request

\- \*\*422 Unprocessable Entity\*\*: Validation error

\- \*\*429 Too Many Requests\*\*: Rate limit exceeded

\- \*\*500 Internal Server Error\*\*: Server error

\- \*\*503 Service Unavailable\*\*: Service not ready



\### Common Errors



\*\*Invalid Platform:\*\*

```json

{

&#x20; "detail": "Validation error",

&#x20; "errors": \[

&#x20;   {

&#x20;     "field": "platform",

&#x20;     "message": "Platform 'facebook' not supported. Supported: \['linkedin', 'twitter', 'bluesky']"

&#x20;   }

&#x20; ]

}

```



\*\*Missing Required Field:\*\*

```json

{

&#x20; "detail": "Validation error",

&#x20; "errors": \[

&#x20;   {

&#x20;     "field": "content",

&#x20;     "message": "field required"

&#x20;   }

&#x20; ]

}

```



\*\*Content Too Long:\*\*

```json

{

&#x20; "detail": "Validation error",

&#x20; "errors": \[

&#x20;   {

&#x20;     "field": "content",

&#x20;     "message": "ensure this value has at most 10000 characters"

&#x20;   }

&#x20; ]

}

```



\## Examples



\### Python



```python

import requests



response = requests.post(

&#x20;   "http://localhost:8000/api/v1/predict",

&#x20;   json={

&#x20;       "content": "Just shipped a new feature!",

&#x20;       "platform": "linkedin",

&#x20;       "has\_link": True

&#x20;   }

)



data = response.json()

print(f"Engagement Score: {data\['engagement\_score']}")

print(f"Confidence: {data\['confidence']}")

```



\### cURL



```bash

curl -X POST http://localhost:8000/api/v1/predict \\

&#x20; -H "Content-Type: application/json" \\

&#x20; -d '{

&#x20;   "content": "Just shipped a new feature!",

&#x20;   "platform": "linkedin",

&#x20;   "has\_link": true

&#x20; }'

```



\### JavaScript



```javascript

const response = await fetch('http://localhost:8000/api/v1/predict', {

&#x20; method: 'POST',

&#x20; headers: {

&#x20;   'Content-Type': 'application/json'

&#x20; },

&#x20; body: JSON.stringify({

&#x20;   content: 'Just shipped a new feature!',

&#x20;   platform: 'linkedin',

&#x20;   has\_link: true

&#x20; })

});



const data = await response.json();

console.log(`Engagement Score: ${data.engagement\_score}`);

```



\## Interactive Documentation



OpenAPI documentation is available at:

\- \*\*Swagger UI:\*\* http://localhost:8000/docs

\- \*\*ReDoc:\*\* http://localhost:8000/redoc

\- \*\*OpenAPI JSON:\*\* http://localhost:8000/openapi.json



\## Support



For issues or questions:

\- GitHub Issues: https://github.com/27manavgandhi/bufferiq/issues

\- Email: 27manavgandhi@gmail.com






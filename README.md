# Pricing Engine Setup and Usage Guide

This guide provides detailed instructions for setting up and using the Pricing Engine, which leverages Amazon competitor data to predict optimal prices for your products, with special support for bulkbuy and groupbuy business models.

## Table of Contents

1. [System Overview](#system-overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Configuration](#configuration)
5. [Database Setup](#database-setup)
6. [Running the Application](#running-the-application)
7. [API Endpoints](#api-endpoints)
8. [Bulkbuy Pricing](#bulkbuy-pricing)
9. [Groupbuy Pricing](#groupbuy-pricing)
10. [Prefect Orchestration](#prefect-orchestration)
11. [Testing](#testing)
12. [Troubleshooting](#troubleshooting)

## System Overview

The Pricing Engine is a comprehensive solution that:

1. Fetches competitor pricing data from Amazon
2. Uses Valkey caching to reduce API calls
3. Predicts optimal prices for your products
4. Supports bulkbuy pricing with quantity-based discounts
5. Supports groupbuy pricing with threshold-based discounts
6. Automatically updates competitor data every 7 days using Prefect

The system consists of several components:

- **Amazon API Client**: Fetches and processes competitor data
- **Valkey Cache**: Caches API responses to reduce calls
- **Database**: Stores product data, competitor data, and pricing information
- **Pricing Model**: Predicts optimal prices based on competitor data
- **Flask API**: Provides endpoints for interacting with the system
- **Prefect Orchestration**: Automates data updates

## Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- Valkey or Redis server
- Amazon Product Advertising API credentials

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/pricing-engine.git
cd pricing-engine
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory with the following variables:

```
# Database Configuration
DB_NAME=bulkmagic
DB_USER=yourusername
DB_PASSWORD=yourpassword
DB_HOST=localhost
DB_PORT=5432

# Valkey/Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Flask Configuration
FLASK_PORT=5000
FLASK_ENV=development

# Amazon API Configuration
AMAZON_ACCESS_KEY=your_access_key_here
AMAZON_SECRET_KEY=your_secret_key_here
AMAZON_PARTNER_TAG=your_partner_tag_here
```

## Database Setup

The system will automatically create the necessary tables when it starts. The database schema includes:

- `products`: Your product catalog
- `amazon_products`: Competitor product data from Amazon
- `amazon_prices`: Historical price data from Amazon
- `predicted_prices`: Price predictions for your products
- `bulk_pricing_tiers`: Quantity-based discount tiers for bulkbuy
- `group_order_thresholds`: Threshold-based discount settings for groupbuy
- `group_orders`: Active and completed group orders

To manually initialize the database:

```bash
python database_schema.py
```

## Running the Application

1. Start the Flask API:

```bash
python app.py
```

2. The API will be available at `http://localhost:5000` (or the port specified in your `.env` file)

## API Endpoints

### Product Management

- `GET /products`: Get all products
- `GET /products/<product_id>`: Get a specific product
- `POST /products`: Create a new product
- `PUT /products/<product_id>`: Update a product

### Competitor Data

- `GET /competitor-prices/<asin>`: Get competitor prices for an ASIN
- `POST /refresh-competitor/<asin>`: Manually refresh competitor data

### Price Prediction

- `GET /predict-price/<product_id>`: Get predicted optimal price for a product
- `GET /price-recommendations/<product_id>`: Get detailed price recommendations

### Bulkbuy Pricing

- `GET /bulk-price/<product_id>?quantity=10`: Get bulk pricing for a specific quantity
- `POST /bulk-pricing-tiers/<product_id>`: Set bulk pricing tiers

### Groupbuy Pricing

- `GET /group-price/<product_id>?current_quantity=15&threshold=25`: Get group pricing
- `POST /group-order-thresholds/<product_id>`: Set group order thresholds
- `GET /group-orders/<product_id>`: Get active group orders for a product
- `POST /group-orders`: Create a new group order
- `POST /group-orders/<group_order_id>/join`: Join a group order by adding quantity

### System Information

- `GET /health`: Health check endpoint
- `GET /stats`: Get system statistics

## Bulkbuy Pricing

Bulkbuy pricing allows customers to receive discounts when purchasing products in bulk. The system calculates discounts based on quantity tiers.

### Setting Up Bulk Pricing Tiers

```bash
curl -X POST http://localhost:5000/bulk-pricing-tiers/YOUR_PRODUCT_ID \
  -H "Content-Type: application/json" \
  -d '{
    "tiers": [
      {"min_quantity": 5, "discount_percentage": 5.0},
      {"min_quantity": 10, "discount_percentage": 10.0},
      {"min_quantity": 25, "discount_percentage": 15.0},
      {"min_quantity": 50, "discount_percentage": 20.0},
      {"min_quantity": 100, "discount_percentage": 25.0}
    ]
  }'
```

### Getting Bulk Price

```bash
curl http://localhost:5000/bulk-price/YOUR_PRODUCT_ID?quantity=10
```

Response:
```json
{
  "product_id": "YOUR_PRODUCT_ID",
  "quantity": 10,
  "base_price": 35.99,
  "bulk_price": 32.39,
  "discount_percentage": 10.0,
  "savings": 3.60,
  "total_price": 323.90
}
```

## Groupbuy Pricing

Groupbuy pricing allows customers to join together to reach a quantity threshold and receive a discount. The system tracks progress toward thresholds and calculates discounts accordingly.

### Setting Up Group Order Thresholds

```bash
curl -X POST http://localhost:5000/group-order-thresholds/YOUR_PRODUCT_ID \
  -H "Content-Type: application/json" \
  -d '{
    "thresholds": [
      {"threshold_quantity": 25, "price_multiplier": 0.9},
      {"threshold_quantity": 50, "price_multiplier": 0.85},
      {"threshold_quantity": 100, "price_multiplier": 0.8}
    ]
  }'
```

### Creating a Group Order

```bash
curl -X POST http://localhost:5000/group-orders \
  -H "Content-Type: application/json" \
  -d '{
    "product_id": "YOUR_PRODUCT_ID",
    "threshold_quantity": 25
  }'
```

Response:
```json
{
  "message": "Group order created successfully",
  "group_order_id": 123,
  "group_order": {
    "id": 123,
    "product_id": "YOUR_PRODUCT_ID",
    "threshold_quantity": 25,
    "current_quantity": 0,
    "status": "active",
    "created_at": "2025-03-18T16:30:00Z"
  }
}
```

### Joining a Group Order

```bash
curl -X POST http://localhost:5000/group-orders/123/join \
  -H "Content-Type: application/json" \
  -d '{
    "quantity": 10
  }'
```

Response:
```json
{
  "message": "Successfully joined group order",
  "group_order_id": 123,
  "product_id": "YOUR_PRODUCT_ID",
  "current_quantity": 10,
  "threshold": 25,
  "remaining": 15,
  "threshold_reached": false,
  "base_price": 35.99,
  "group_price": 34.19,
  "status": "active"
}
```

### Getting Group Price

```bash
curl http://localhost:5000/group-price/YOUR_PRODUCT_ID?current_quantity=15&threshold=25
```

Response:
```json
{
  "product_id": "YOUR_PRODUCT_ID",
  "current_quantity": 15,
  "threshold": 25,
  "remaining": 10,
  "base_price": 35.99,
  "group_price": 33.29,
  "savings_per_unit": 2.70,
  "total_savings": 40.50,
  "group_complete": false
}
```

## Prefect Orchestration

The system uses Prefect to automatically update competitor data every 7 days. This ensures that your pricing remains competitive without requiring manual intervention.

### Starting the Prefect Orchestration

```bash
python prefect_orchestration.py
```

### Prefect Flows

1. `update_competitor_prices`: Runs every 7 days to update competitor data
2. `process_group_orders`: Runs daily to process completed group orders

## Testing

The system includes comprehensive tests for all components. To run the tests:

```bash
python test_implementation.py
```

The tests cover:
- Amazon API client
- Valkey caching system
- Database operations
- Pricing model
- Flask API endpoints
- Bulkbuy and groupbuy functionality

## Troubleshooting

### Common Issues

1. **Database Connection Errors**
   - Check your database credentials in the `.env` file
   - Ensure PostgreSQL is running

2. **Valkey/Redis Connection Errors**
   - Check your Valkey/Redis configuration in the `.env` file
   - Ensure Valkey/Redis server is running

3. **Amazon API Errors**
   - Verify your Amazon API credentials in the `.env` file
   - Check API request limits and throttling

4. **Missing Predictions**
   - Ensure products have a valid `competitor_asin` set
   - Check if competitor data is available for the ASIN

### Logs

The system logs information to the console. For more detailed logging, you can adjust the log level in each module.

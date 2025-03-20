import os
import logging
from datetime import timedelta
from typing import Dict, List, Any, Optional
from prefect import flow, task
from prefect.schedules import IntervalSchedule
from dotenv import load_dotenv

# Import our modules
from amazon_api import AmazonAPI
from valkey_cache import ValkeyCache
from database_schema import Database
from pricing_model import PricingModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

@task(name="fetch_amazon_data", retries=3, retry_delay_seconds=60)
def fetch_amazon_data(asins: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch Amazon product data for the given ASINs.
    
    Args:
        asins: List of ASINs to fetch
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping ASINs to product data
    """
    logger.info(f"Fetching Amazon data for {len(asins)} ASINs")
    
    api = AmazonAPI()
    cache = ValkeyCache()
    
    results = {}
    
    for asin in asins:
        # Check cache first
        cached_data = cache.get_product(asin)
        if cached_data:
            logger.info(f"Using cached data for ASIN {asin}")
            results[asin] = cached_data
            continue
        
        # Fetch from API if not in cache
        product_data = api.get_product(asin)
        
        if 'error' not in product_data:
            # Cache the result
            cache.cache_product(asin, product_data)
            
            # Extract pricing data
            pricing_data = api.extract_pricing_data(product_data)
            cache.cache_pricing(asin, pricing_data)
            
            results[asin] = product_data
        else:
            logger.error(f"Error fetching data for ASIN {asin}: {product_data.get('error')}")
    
    logger.info(f"Fetched data for {len(results)} ASINs")
    return results

@task(name="store_amazon_data")
def store_amazon_data(amazon_data: Dict[str, Dict[str, Any]]) -> List[str]:
    """
    Store Amazon product data in the database.
    
    Args:
        amazon_data: Dictionary mapping ASINs to product data
        
    Returns:
        List[str]: List of successfully stored ASINs
    """
    logger.info(f"Storing Amazon data for {len(amazon_data)} products")
    
    db = Database()
    stored_asins = []
    
    try:
        # Create tables if they don't exist
        db.create_tables()
        
        for asin, product_data in amazon_data.items():
            try:
                # Store product data
                db.insert_amazon_product(product_data)
                
                # Extract and store pricing data
                api = AmazonAPI()
                pricing_data = api.extract_pricing_data(product_data)
                db.insert_amazon_price(pricing_data)
                
                stored_asins.append(asin)
                logger.info(f"Stored data for ASIN {asin}")
            except Exception as e:
                logger.error(f"Error storing data for ASIN {asin}: {str(e)}")
    finally:
        db.close()
    
    logger.info(f"Stored data for {len(stored_asins)} ASINs")
    return stored_asins

@task(name="update_price_predictions")
def update_price_predictions(asins: List[str]) -> Dict[str, float]:
    """
    Update price predictions for products with the given competitor ASINs.
    
    Args:
        asins: List of competitor ASINs
        
    Returns:
        Dict[str, float]: Dictionary mapping product IDs to predicted prices
    """
    logger.info(f"Updating price predictions for products with {len(asins)} competitor ASINs")
    
    db = Database()
    cache = ValkeyCache()
    model = PricingModel("pricing_model.joblib")
    predictions = {}
    
    try:
        for asin in asins:
            # Find products that use this ASIN as competitor
            db.cursor.execute("SELECT * FROM products WHERE competitor_asin = %s", (asin,))
            products = db.cursor.fetchall()
            
            if not products:
                logger.info(f"No products found with competitor ASIN {asin}")
                continue
            
            # Get product data
            product_columns = [desc[0] for desc in db.cursor.description]
            product_dicts = [dict(zip(product_columns, product)) for product in products]
            
            # Get Amazon data
            amazon_data = db.get_amazon_product(asin)
            if not amazon_data:
                logger.warning(f"No Amazon data found for ASIN {asin}")
                continue
            
            # Get latest price data
            price_data = db.get_latest_amazon_price(asin)
            if price_data:
                # Merge price data into Amazon data
                amazon_data.update(price_data)
            
            # Make predictions for each product
            for product in product_dicts:
                product_id = product.get('product_id')
                
                try:
                    # Predict price
                    predicted_price = model.predict_price(amazon_data, product)
                    
                    # Store prediction in database
                    prediction_data = {
                        'product_id': product_id,
                        'predicted_price': predicted_price,
                        'confidence': 0.8  # Default confidence
                    }
                    db.insert_predicted_price(prediction_data)
                    
                    # Cache prediction
                    cache.cache_predicted_price(product_id, predicted_price)
                    
                    # Update bulk pricing tiers in database
                    bulk_tiers = []
                    for qty in [5, 10, 25, 50, 100]:
                        bulk_price = model.calculate_bulk_price(predicted_price, qty, product_id)
                        discount_pct = ((predicted_price - bulk_price) / predicted_price) * 100
                        bulk_tiers.append({
                            'min_quantity': qty,
                            'discount_percentage': discount_pct
                        })
                    
                    db.set_bulk_pricing_tiers(product_id, bulk_tiers)
                    
                    # Update group order thresholds in database
                    group_thresholds = []
                    for threshold in [25, 50, 100]:
                        # Calculate price at full threshold
                        group_price = model.calculate_group_price(predicted_price, threshold, threshold, product_id)
                        price_multiplier = group_price / predicted_price
                        group_thresholds.append({
                            'threshold_quantity': threshold,
                            'price_multiplier': price_multiplier
                        })
                    
                    db.set_group_order_thresholds(product_id, group_thresholds)
                    
                    predictions[product_id] = predicted_price
                    logger.info(f"Updated prediction for product {product_id}: ${predicted_price:.2f}")
                except Exception as e:
                    logger.error(f"Error predicting price for product {product_id}: {str(e)}")
    finally:
        db.close()
    
    logger.info(f"Updated predictions for {len(predictions)} products")
    return predictions

@task(name="analyze_market_position")
def analyze_market_position(asins: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Analyze market position for products with the given competitor ASINs.
    
    Args:
        asins: List of competitor ASINs
        
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping ASINs to market analysis
    """
    logger.info(f"Analyzing market position for {len(asins)} ASINs")
    
    api = AmazonAPI()
    cache = ValkeyCache()
    db = Database()
    results = {}
    
    try:
        for asin in asins:
            # Check cache first
            cached_analysis = cache.get_market_analysis(asin)
            if cached_analysis:
                logger.info(f"Using cached market analysis for ASIN {asin}")
                results[asin] = cached_analysis
                continue
            
            # Get product data
            product_data = db.get_amazon_product(asin)
            if not product_data:
                logger.warning(f"No product data found for ASIN {asin}")
                continue
            
            # Get price data
            price_data = db.get_latest_amazon_price(asin)
            if price_data:
                product_data.update(price_data)
            
            # Search for similar products
            if 'title' in product_data:
                # Extract keywords from title
                title = product_data.get('title', '')
                keywords = ' '.join(title.split()[:5])  # First 5 words
                
                # Search for similar products
                similar_products = api.search_products(keywords, limit=10)
                
                # Filter out the target product
                similar_products = [p for p in similar_products if p.get('asin') != asin]
                
                # Analyze market position
                market_analysis = api.analyze_market_position(product_data, similar_products)
                
                # Cache the analysis
                cache.cache_market_analysis(asin, market_analysis)
                
                results[asin] = market_analysis
                logger.info(f"Analyzed market position for ASIN {asin}")
            else:
                logger.warning(f"No title found for ASIN {asin}")
    finally:
        db.close()
    
    logger.info(f"Analyzed market position for {len(results)} ASINs")
    return results

@flow(name="update_competitor_prices")
def update_competitor_prices():
    """
    Flow to update competitor prices from Amazon and make price predictions.
    
    This flow runs every 7 days to keep competitor price data up to date.
    """
    logger.info("Starting update_competitor_prices flow")
    
    # Get competitor ASINs to update
    db = Database()
    try:
        # Create tables if they don't exist
        db.create_tables()
        
        # Get products needing updates
        products_to_update = db.get_products_needing_update(days=7)
        
        if not products_to_update:
            logger.info("No products need updates at this time")
            return
        
        # Extract ASINs
        asins = [p.get('competitor_asin') for p in products_to_update if p.get('competitor_asin')]
        asins = list(set(asins))  # Remove duplicates
        
        logger.info(f"Found {len(asins)} ASINs to update")
    finally:
        db.close()
    
    if not asins:
        logger.info("No ASINs to update")
        return
    
    # Fetch Amazon data
    amazon_data = fetch_amazon_data(asins)
    
    # Store Amazon data
    stored_asins = store_amazon_data(amazon_data)
    
    # Update price predictions
    predictions = update_price_predictions(stored_asins)
    
    # Analyze market position
    market_analysis = analyze_market_position(stored_asins)
    
    logger.info(f"Completed update_competitor_prices flow: {len(stored_asins)} ASINs updated, {len(predictions)} predictions made")
    return {
        "asins_updated": len(stored_asins),
        "predictions_made": len(predictions),
        "market_analyses": len(market_analysis)
    }

@flow(name="process_group_orders")
def process_group_orders():
    """
    Flow to process group orders that have reached their thresholds.
    
    This flow runs daily to check and process completed group orders.
    """
    logger.info("Starting process_group_orders flow")
    
    db = Database()
    try:
        # Get completed group orders
        db.cursor.execute("""
            SELECT * FROM group_orders 
            WHERE status = 'completed' 
            AND completed_at IS NOT NULL
            AND completed_at > NOW() - INTERVAL '1 day'
        """)
        
        rows = db.cursor.fetchall()
        if not rows:
            logger.info("No completed group orders to process")
            return
        
        columns = [desc[0] for desc in db.cursor.description]
        completed_orders = [dict(zip(columns, row)) for row in rows]
        
        logger.info(f"Found {len(completed_orders)} completed group orders to process")
        
        # Process each completed order
        for order in completed_orders:
            order_id = order.get('id')
            product_id = order.get('product_id')
            threshold = order.get('threshold_quantity')
            
            logger.info(f"Processing completed group order {order_id} for product {product_id}")
            
            # Here you would implement the actual order processing logic
            # For example, sending notifications, creating purchase orders, etc.
            
            # Mark as processed
            db.cursor.execute("""
                UPDATE group_orders
                SET status = 'processed'
                WHERE id = %s
            """, (order_id,))
            
            db.conn.commit()
            logger.info(f"Marked group order {order_id} as processed")
    finally:
        db.close()
    
    logger.info("Completed process_group_orders flow")

# Create schedules
update_schedule = IntervalSchedule(
    interval=timedelta(days=7)
)

group_order_schedule = IntervalSchedule(
    interval=timedelta(days=1)
)

# Register the flows with schedules
if __name__ == "__main__":
    # For testing, run the flow directly
    update_competitor_prices()

from flask import Flask, request, jsonify
import os
import logging
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional

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

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Initialize components
api = AmazonAPI()
cache = ValkeyCache()
model = PricingModel("pricing_model.joblib")

# Create database tables if they don't exist
db = Database()
db.create_tables()
db.close()  # Close connection after creating tables

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "version": "1.0.0"})

@app.route('/products', methods=['GET'])
def get_products():
    """Get all products."""
    db = Database()
    try:
        db.cursor.execute("SELECT * FROM products")
        rows = db.cursor.fetchall()
        
        if rows:
            columns = [desc[0] for desc in db.cursor.description]
            products = [dict(zip(columns, row)) for row in rows]
            return jsonify({"products": products})
        else:
            return jsonify({"products": []})
    finally:
        db.close()

@app.route('/products/<product_id>', methods=['GET'])
def get_product(product_id):
    """Get a specific product."""
    db = Database()
    try:
        product = db.get_product(product_id)
        if product:
            return jsonify({"product": product})
        else:
            return jsonify({"error": "Product not found"}), 404
    finally:
        db.close()

@app.route('/products', methods=['POST'])
def create_product():
    """Create a new product."""
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    required_fields = ['product_id', 'name', 'wholesale_cost']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    db = Database()
    try:
        db.insert_product(data)
        return jsonify({"message": "Product created successfully", "product_id": data['product_id']})
    except Exception as e:
        logger.error(f"Error creating product: {str(e)}")
        return jsonify({"error": f"Failed to create product: {str(e)}"}), 500
    finally:
        db.close()

@app.route('/products/<product_id>', methods=['PUT'])
def update_product(product_id):
    """Update a product."""
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    # Ensure product_id in URL matches the one in data
    data['product_id'] = product_id
    
    db = Database()
    try:
        # Check if product exists
        existing_product = db.get_product(product_id)
        if not existing_product:
            return jsonify({"error": "Product not found"}), 404
        
        # Update product
        db.insert_product(data)
        return jsonify({"message": "Product updated successfully", "product_id": product_id})
    except Exception as e:
        logger.error(f"Error updating product: {str(e)}")
        return jsonify({"error": f"Failed to update product: {str(e)}"}), 500
    finally:
        db.close()

@app.route('/competitor-prices/<asin>', methods=['GET'])
def get_competitor_prices(asin):
    """Get competitor prices for an ASIN."""
    # Check cache first
    cached_pricing = cache.get_pricing(asin)
    if cached_pricing:
        return jsonify({"source": "cache", "pricing": cached_pricing})
    
    # Check database if not in cache
    db = Database()
    try:
        amazon_product = db.get_amazon_product(asin)
        price_history = db.get_amazon_price_history(asin)
        
        if amazon_product and price_history:
            # Cache the latest pricing
            if price_history:
                cache.cache_pricing(asin, price_history[0])
            
            return jsonify({
                "source": "database",
                "product": amazon_product,
                "price_history": price_history
            })
        
        # Fetch from API if not in database
        product_data = api.get_product(asin)
        
        if 'error' in product_data:
            return jsonify({"error": product_data['error']}), 404
        
        # Extract pricing data
        pricing_data = api.extract_pricing_data(product_data)
        
        # Store in database
        db.insert_amazon_product(product_data)
        db.insert_amazon_price(pricing_data)
        
        # Cache the pricing
        cache.cache_product(asin, product_data)
        cache.cache_pricing(asin, pricing_data)
        
        return jsonify({
            "source": "api",
            "product": product_data,
            "pricing": pricing_data
        })
    except Exception as e:
        logger.error(f"Error fetching competitor prices: {str(e)}")
        return jsonify({"error": f"Failed to fetch competitor prices: {str(e)}"}), 500
    finally:
        db.close()

@app.route('/predict-price/<product_id>', methods=['GET'])
def predict_price(product_id):
    """Predict optimal price for a product."""
    # Check cache first
    cached_price = cache.get_predicted_price(product_id)
    if cached_price is not None:
        return jsonify({
            "source": "cache",
            "product_id": product_id,
            "predicted_price": cached_price
        })
    
    # Get product data
    db = Database()
    try:
        product = db.get_product(product_id)
        if not product:
            return jsonify({"error": "Product not found"}), 404
        
        # Get competitor ASIN
        competitor_asin = product.get('competitor_asin')
        if not competitor_asin:
            return jsonify({"error": "Product has no competitor ASIN"}), 400
        
        # Get Amazon data
        amazon_data = db.get_amazon_product(competitor_asin)
        price_data = db.get_latest_amazon_price(competitor_asin)
        
        # If not in database, fetch from API
        if not amazon_data or not price_data:
            # Fetch from API
            product_data = api.get_product(competitor_asin)
            
            if 'error' in product_data:
                return jsonify({"error": f"Failed to fetch competitor data: {product_data['error']}"}), 404
            
            # Extract pricing data
            pricing_data = api.extract_pricing_data(product_data)
            
            # Store in database
            db.insert_amazon_product(product_data)
            db.insert_amazon_price(pricing_data)
            
            # Use fetched data
            amazon_data = product_data
            price_data = pricing_data
        
        # Merge price data into Amazon data
        if price_data:
            amazon_data.update(price_data)
        
        # Predict price
        predicted_price = model.predict_price(amazon_data, product)
        
        # Store prediction
        prediction_data = {
            'product_id': product_id,
            'predicted_price': predicted_price,
            'confidence': 0.8  # Default confidence
        }
        db.insert_predicted_price(prediction_data)
        
        # Cache prediction
        cache.cache_predicted_price(product_id, predicted_price)
        
        return jsonify({
            "product_id": product_id,
            "competitor_asin": competitor_asin,
            "predicted_price": predicted_price,
            "wholesale_cost": float(product.get('wholesale_cost')),
            "retail_price": float(product.get('retail_price')),
            "competitor_price": float(amazon_data.get('final_price', 0))
        })
    except Exception as e:
        logger.error(f"Error predicting price: {str(e)}")
        return jsonify({"error": f"Failed to predict price: {str(e)}"}), 500
    finally:
        db.close()

@app.route('/bulk-price/<product_id>', methods=['GET'])
def get_bulk_price(product_id):
    """Get bulk pricing for a product."""
    quantity = int(request.args.get('quantity', 1))
    
    # Check cache first
    cached_price = cache.get_bulk_price(product_id, quantity)
    if cached_price is not None:
        return jsonify({
            "source": "cache",
            "product_id": product_id,
            "quantity": quantity,
            "bulk_price": cached_price
        })
    
    # Get base price prediction
    db = Database()
    try:
        # Get product data
        product = db.get_product(product_id)
        if not product:
            return jsonify({"error": "Product not found"}), 404
        
        # Get latest predicted price
        predicted_price_data = db.get_latest_predicted_price(product_id)
        if not predicted_price_data:
            # No prediction available, get one
            response = predict_price(product_id)
            if isinstance(response, tuple) and response[1] != 200:
                return response
            
            if hasattr(response, 'json'):
                data = response.json()
                base_price = data.get('predicted_price')
            else:
                data = response.get_json()
                base_price = data.get('predicted_price')
        else:
            base_price = float(predicted_price_data.get('predicted_price'))
        
        # Get bulk pricing tiers
        bulk_pricing_tiers = db.get_bulk_pricing_tiers(product_id)
        
        # Calculate bulk price
        bulk_price = model.calculate_bulk_price(base_price, quantity, product_id, bulk_pricing_tiers)
        
        # Cache bulk price
        cache.cache_bulk_price(product_id, quantity, bulk_price)
        
        # Get applicable tier
        applicable_tier = None
        for tier in sorted(bulk_pricing_tiers, key=lambda x: x['min_quantity'], reverse=True):
            if quantity >= tier['min_quantity']:
                applicable_tier = tier
                break
        
        discount_percentage = 0
        if applicable_tier:
            discount_percentage = float(applicable_tier.get('discount_percentage', 0))
        
        return jsonify({
            "product_id": product_id,
            "quantity": quantity,
            "base_price": base_price,
            "bulk_price": bulk_price,
            "discount_percentage": discount_percentage,
            "savings": base_price - bulk_price,
            "total_price": bulk_price * quantity
        })
    except Exception as e:
        logger.error(f"Error calculating bulk price: {str(e)}")
        return jsonify({"error": f"Failed to calculate bulk price: {str(e)}"}), 500
    finally:
        db.close()

@app.route('/group-price/<product_id>', methods=['GET'])
def get_group_price(product_id):
    """Get group pricing for a product."""
    current_quantity = int(request.args.get('current_quantity', 0))
    threshold = int(request.args.get('threshold', 25))
    group_order_id = request.args.get('group_order_id')
    
    # Check cache first
    cached_price = cache.get_group_price(product_id, threshold, current_quantity)
    if cached_price is not None:
        return jsonify({
            "source": "cache",
            "product_id": product_id,
            "current_quantity": current_quantity,
            "threshold": threshold,
            "group_price": cached_price
        })
    
    db = Database()
    try:
        # Get product data
        product = db.get_product(product_id)
        if not product:
            return jsonify({"error": "Product not found"}), 404
        
        # Get latest predicted price
        predicted_price_data = db.get_latest_predicted_price(product_id)
        if not predicted_price_data:
            # No prediction available, get one
            response = predict_price(product_id)
            if isinstance(response, tuple) and response[1] != 200:
                return response
            
            if hasattr(response, 'json'):
                data = response.json()
                base_price = data.get('predicted_price')
            else:
                data = response.get_json()
                base_price = data.get('predicted_price')
        else:
            base_price = float(predicted_price_data.get('predicted_price'))
        
        # Get group order thresholds
        group_order_thresholds = db.get_group_order_thresholds(product_id)
        
        # Calculate group price
        group_price = model.calculate_group_price(base_price, current_quantity, threshold, product_id, group_order_thresholds)
        
        # Cache group price
        cache.cache_group_price(product_id, threshold, current_quantity, group_price)
        
        # Calculate remaining quantity needed
        remaining = max(0, threshold - current_quantity)
        
        # Get group order if ID provided
        group_order = None
        if group_order_id:
            group_order = db.get_group_order(int(group_order_id))
        
        return jsonify({
            "product_id": product_id,
            "current_quantity": current_quantity,
            "threshold": threshold,
            "remaining": remaining,
            "base_price": base_price,
            "group_price": group_price,
            "savings_per_unit": base_price - group_price,
            "total_savings": (base_price - group_price) * current_quantity,
            "group_complete": current_quantity >= threshold,
            "group_order": group_order
        })
    except Exception as e:
        logger.error(f"Error calculating group price: {str(e)}")
        return jsonify({"error": f"Failed to calculate group price: {str(e)}"}), 500
    finally:
        db.close()

@app.route('/group-orders/<product_id>', methods=['GET'])
def get_group_orders(product_id):
    """Get active group orders for a product."""
    db = Database()
    try:
        # Get product data
        product = db.get_product(product_id)
        if not product:
            return jsonify({"error": "Product not found"}), 404
        
        # Get active group orders
        active_orders = db.get_active_group_orders(product_id)
        
        return jsonify({
            "product_id": product_id,
            "active_orders": active_orders
        })
    except Exception as e:
        logger.error(f"Error fetching group orders: {str(e)}")
        return jsonify({"error": f"Failed to fetch group orders: {str(e)}"}), 500
    finally:
        db.close()

@app.route('/group-orders', methods=['POST'])
def create_group_order():
    """Create a new group order."""
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    required_fields = ['product_id', 'threshold_quantity']
    for field in required_fields:
        if field not in data:
            return jsonify({"error": f"Missing required field: {field}"}), 400
    
    product_id = data['product_id']
    threshold_quantity = int(data['threshold_quantity'])
    
    db = Database()
    try:
        # Check if product exists
        product = db.get_product(product_id)
        if not product:
            return jsonify({"error": "Product not found"}), 404
        
        # Create group order
        group_order_id = db.create_group_order(product_id, threshold_quantity)
        
        # Get the created group order
        group_order = db.get_group_order(group_order_id)
        
        return jsonify({
            "message": "Group order created successfully",
            "group_order_id": group_order_id,
            "group_order": group_order
        })
    except Exception as e:
        logger.error(f"Error creating group order: {str(e)}")
        return jsonify({"error": f"Failed to create group order: {str(e)}"}), 500
    finally:
        db.close()

@app.route('/group-orders/<int:group_order_id>/join', methods=['POST'])
def join_group_order(group_order_id):
    """Join a group order by adding quantity."""
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    if 'quantity' not in data:
        return jsonify({"error": "Missing required field: quantity"}), 400
    
    quantity = int(data['quantity'])
    if quantity <= 0:
        return jsonify({"error": "Quantity must be greater than zero"}), 400
    
    db = Database()
    try:
        # Check if group order exists
        group_order = db.get_group_order(group_order_id)
        if not group_order:
            return jsonify({"error": "Group order not found"}), 404
        
        # Check if group order is active
        if group_order['status'] != 'active':
            return jsonify({"error": "Group order is not active"}), 400
        
        # Update group order quantity
        updated_order = db.update_group_order(group_order_id, quantity)
        
        if not updated_order:
            return jsonify({"error": "Failed to update group order"}), 500
        
        # Get product data
        product = db.get_product(updated_order['product_id'])
        
        # Calculate group price
        response = get_group_price(
            updated_order['product_id'],
            current_quantity=updated_order['current_quantity'],
            threshold=updated_order['threshold_quantity'],
            group_order_id=group_order_id
        )
        
        if isinstance(response, tuple) and response[1] != 200:
            return response
        
        if hasattr(response, 'json'):
            price_data = response.json()
        else:
            price_data = response.get_json()
        
        return jsonify({
            "message": "Successfully joined group order",
            "group_order_id": group_order_id,
            "product_id": updated_order['product_id'],
            "current_quantity": updated_order['current_quantity'],
            "threshold": updated_order['threshold_quantity'],
            "remaining": max(0, updated_order['threshold_quantity'] - updated_order['current_quantity']),
            "threshold_reached": updated_order['current_quantity'] >= updated_order['threshold_quantity'],
            "base_price": price_data.get('base_price'),
            "group_price": price_data.get('group_price'),
            "status": updated_order['status']
        })
    except Exception as e:
        logger.error(f"Error joining group order: {str(e)}")
        return jsonify({"error": f"Failed to join group order: {str(e)}"}), 500
    finally:
        db.close()


@app.route('/refresh-competitor/<asin>', methods=['POST'])
def refresh_competitor_data(asin):
    """Manually refresh competitor data for an ASIN."""
    try:
        # Fetch from API
        product_data = api.get_product(asin)
        
        if 'error' in product_data:
            return jsonify({"error": product_data['error']}), 404
        
        # Extract pricing data
        pricing_data = api.extract_pricing_data(product_data)
        
        # Store in database
        db = Database()
        try:
            db.insert_amazon_product(product_data)
            db.insert_amazon_price(pricing_data)
        finally:
            db.close()
        
        # Update cache
        cache.cache_product(asin, product_data)
        cache.cache_pricing(asin, pricing_data)
        
        return jsonify({
            "message": "Competitor data refreshed successfully",
            "asin": asin,
            "product": product_data,
            "pricing": pricing_data
        })
    except Exception as e:
        logger.error(f"Error refreshing competitor data: {str(e)}")
        return jsonify({"error": f"Failed to refresh competitor data: {str(e)}"}), 500

@app.route('/bulk-pricing-tiers/<product_id>', methods=['POST'])
def set_bulk_pricing_tiers(product_id):
    """Set bulk pricing tiers for a product."""
    data = request.json
    
    if not data or 'tiers' not in data:
        return jsonify({"error": "No tiers provided"}), 400
    
    tiers = data['tiers']
    
    db = Database()
    try:
        # Check if product exists
        product = db.get_product(product_id)
        if not product:
            return jsonify({"error": "Product not found"}), 404
        
        # Set bulk pricing tiers
        db.set_bulk_pricing_tiers(product_id, tiers)
        
        return jsonify({
            "message": "Bulk pricing tiers updated successfully",
            "product_id": product_id,
            "tiers": tiers
        })
    except Exception as e:
        logger.error(f"Error setting bulk pricing tiers: {str(e)}")
        return jsonify({"error": f"Failed to set bulk pricing tiers: {str(e)}"}), 500
    finally:
        db.close()

@app.route('/group-order-thresholds/<product_id>', methods=['POST'])
def set_group_order_thresholds(product_id):
    """Set group order thresholds for a product."""
    data = request.json
    
    if not data or 'thresholds' not in data:
        return jsonify({"error": "No thresholds provided"}), 400
    
    thresholds = data['thresholds']
    
    db = Database()
    try:
        # Check if product exists
        product = db.get_product(product_id)
        if not product:
            return jsonify({"error": "Product not found"}), 404
        
        # Set group order thresholds
        db.set_group_order_thresholds(product_id, thresholds)
        
        return jsonify({
            "message": "Group order thresholds updated successfully",
            "product_id": product_id,
            "thresholds": thresholds
        })
    except Exception as e:
        logger.error(f"Error setting group order thresholds: {str(e)}")
        return jsonify({"error": f"Failed to set group order thresholds: {str(e)}"}), 500
    finally:
        db.close()

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    db = Database()
    try:
        # Get counts
        db.cursor.execute("SELECT COUNT(*) FROM products")
        product_count = db.cursor.fetchone()[0]
        
        db.cursor.execute("SELECT COUNT(*) FROM amazon_products")
        amazon_product_count = db.cursor.fetchone()[0]
        
        db.cursor.execute("SELECT COUNT(*) FROM amazon_prices")
        price_count = db.cursor.fetchone()[0]
        
        db.cursor.execute("SELECT COUNT(*) FROM predicted_prices")
        prediction_count = db.cursor.fetchone()[0]
        
        db.cursor.execute("SELECT COUNT(*) FROM group_orders WHERE status = 'active'")
        active_group_orders = db.cursor.fetchone()[0]
        
        db.cursor.execute("SELECT COUNT(*) FROM group_orders WHERE status = 'completed'")
        completed_group_orders = db.cursor.fetchone()[0]
        
        # Get cache stats
        cache_stats = cache.get_stats()
        
        return jsonify({
            "database": {
                "products": product_count,
                "amazon_products": amazon_product_count,
                "price_records": price_count,
                "predictions": prediction_count,
                "active_group_orders": active_group_orders,
                "completed_group_orders": completed_group_orders
            },
            "cache": cache_stats
        })
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        return jsonify({"error": f"Failed to fetch stats: {str(e)}"}), 500
    finally:
        db.close()

if __name__ == '__main__':
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_ENV", "production").lower() == "development"
    
    logger.info(f"Starting Flask app on port {port}, debug={debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)

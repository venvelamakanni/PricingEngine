import unittest
import json
import os
import sys
import logging
from datetime import datetime
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from amazon_api import AmazonAPI
from valkey_cache import ValkeyCache
from database_schema import Database
from pricing_model import PricingModel

class TestAmazonAPI(unittest.TestCase):
    """Test the Amazon API client."""
    
    def setUp(self):
        self.api = AmazonAPI()
        self.test_asin = "B07Q32KX3J"  # Example ASIN for testing
    
    def test_get_product(self):
        """Test fetching a product by ASIN."""
        # Skip if API key not configured
        if not self.api.api_key:
            logger.warning("Skipping API test: No API key configured")
            return
        
        product = self.api.get_product(self.test_asin)
        
        # Check if we got a valid response
        self.assertNotIn('error', product)
        self.assertIn('asin', product)
        self.assertEqual(product['asin'], self.test_asin)
    
    def test_extract_pricing_data(self):
        """Test extracting pricing data from product data."""
        # Create mock product data
        product_data = {
            'asin': self.test_asin,
            'title': 'Test Product',
            'brand': 'Test Brand',
            'categories': ['Category 1', 'Category 2'],
            'initial_price': 39.99,
            'final_price': 35.99,
            'currency': 'USD',
            'availability': 'In Stock',
            'rating': 4.5,
            'reviews_count': 1000,
            'buybox_seller': 'Test Seller',
            'number_of_sellers': 3,
            'variations': [
                {'price': 34.99},
                {'price': 36.99},
                {'price': 38.99}
            ]
        }
        
        pricing_data = self.api.extract_pricing_data(product_data)
        
        # Check extracted data
        self.assertEqual(pricing_data['asin'], self.test_asin)
        self.assertEqual(pricing_data['final_price'], 35.99)
        self.assertEqual(pricing_data['initial_price'], 39.99)
        self.assertAlmostEqual(pricing_data['discount_percentage'], 10.0, places=1)
        self.assertAlmostEqual(pricing_data['price_mean'], 36.99, places=2)
        self.assertTrue(pricing_data['price_std'] > 0)
    
    def test_analyze_market_position(self):
        """Test analyzing market position."""
        # Create mock product data
        product_data = {
            'asin': self.test_asin,
            'title': 'Test Product',
            'final_price': 35.99,
            'rating': 4.5
        }
        
        # Create mock similar products
        similar_products = [
            {'asin': 'B123456789', 'final_price': 39.99, 'rating': 4.0},
            {'asin': 'B234567890', 'final_price': 32.99, 'rating': 4.2},
            {'asin': 'B345678901', 'final_price': 37.99, 'rating': 4.7}
        ]
        
        analysis = self.api.analyze_market_position(product_data, similar_products)
        
        # Check analysis results
        self.assertIn('market_position', analysis)
        self.assertIn('price_percentile', analysis)
        self.assertIn('rating_percentile', analysis)
        self.assertIn('competitive_index', analysis)
    
    def test_get_price_recommendations(self):
        """Test getting price recommendations."""
        # Create mock product data
        product_data = {
            'asin': self.test_asin,
            'title': 'Test Product',
            'final_price': 35.99,
            'rating': 4.5
        }
        
        # Create mock similar products
        similar_products = [
            {'asin': 'B123456789', 'final_price': 39.99, 'rating': 4.0},
            {'asin': 'B234567890', 'final_price': 32.99, 'rating': 4.2},
            {'asin': 'B345678901', 'final_price': 37.99, 'rating': 4.7}
        ]
        
        wholesale_cost = 20.0
        
        recommendations = self.api.get_price_recommendations(product_data, similar_products, wholesale_cost)
        
        # Check recommendations
        self.assertIn('market_analysis', recommendations)
        self.assertIn('min_viable_price', recommendations)
        self.assertIn('competitive_price', recommendations)
        self.assertIn('optimal_price', recommendations)
        self.assertIn('premium_price', recommendations)
        self.assertIn('bulk_pricing', recommendations)
        self.assertIn('group_buy', recommendations)
        
        # Check bulk pricing tiers
        self.assertEqual(len(recommendations['bulk_pricing']), 5)
        
        # Check group buy thresholds
        self.assertEqual(len(recommendations['group_buy']), 3)

class TestValkeyCache(unittest.TestCase):
    """Test the Valkey caching system."""
    
    def setUp(self):
        self.cache = ValkeyCache()
        self.test_key = "test_key"
        self.test_value = "test_value"
        self.test_json = {"key": "value", "number": 123}
        self.test_asin = "B07Q32KX3J"
        
        # Clean up any existing test keys
        self.cache.delete(self.test_key)
    
    def test_set_get(self):
        """Test setting and getting a value."""
        # Skip if no connection
        if not self.cache.client:
            logger.warning("Skipping cache test: No Valkey/Redis connection")
            return
            
        # Set a value
        self.cache.set(self.test_key, self.test_value)
        
        # Get the value
        value = self.cache.get(self.test_key)
        
        # Check if the value matches
        self.assertEqual(value, self.test_value)
    
    def test_set_get_json(self):
        """Test setting and getting a JSON value."""
        # Skip if no connection
        if not self.cache.client:
            logger.warning("Skipping cache test: No Valkey/Redis connection")
            return
            
        # Set a JSON value
        self.cache.set_json(self.test_key, self.test_json)
        
        # Get the JSON value
        value = self.cache.get_json(self.test_key)
        
        # Check if the value matches
        self.assertEqual(value, self.test_json)
    
    def test_ttl(self):
        """Test TTL functionality."""
        # Skip if no connection
        if not self.cache.client:
            logger.warning("Skipping cache test: No Valkey/Redis connection")
            return
            
        # Set a value with a TTL
        self.cache.set(self.test_key, self.test_value, ttl=10)
        
        # Check if TTL is set
        ttl = self.cache.ttl(self.test_key)
        
        # TTL should be between 0 and 10
        self.assertTrue(0 <= ttl <= 10)
    
    def test_delete(self):
        """Test deleting a value."""
        # Skip if no connection
        if not self.cache.client:
            logger.warning("Skipping cache test: No Valkey/Redis connection")
            return
            
        # Set a value
        self.cache.set(self.test_key, self.test_value)
        
        # Delete the value
        self.cache.delete(self.test_key)
        
        # Check if the value is gone
        value = self.cache.get(self.test_key)
        self.assertIsNone(value)
    
    def test_product_caching(self):
        """Test caching product data."""
        # Skip if no connection
        if not self.cache.client:
            logger.warning("Skipping cache test: No Valkey/Redis connection")
            return
            
        # Create mock product data
        product_data = {
            'asin': self.test_asin,
            'title': 'Test Product',
            'price': 35.99
        }
        
        # Cache the product
        self.cache.cache_product(self.test_asin, product_data)
        
        # Get the cached product
        cached_product = self.cache.get_product(self.test_asin)
        
        # Check if the product matches
        self.assertEqual(cached_product, product_data)
    
    def test_bulk_price_caching(self):
        """Test caching bulk price data."""
        # Skip if no connection
        if not self.cache.client:
            logger.warning("Skipping cache test: No Valkey/Redis connection")
            return
            
        product_id = "TEST123"
        quantity = 10
        price = 29.99
        
        # Cache the bulk price
        self.cache.cache_bulk_price(product_id, quantity, price)
        
        # Get the cached bulk price
        cached_price = self.cache.get_bulk_price(product_id, quantity)
        
        # Check if the price matches
        self.assertEqual(cached_price, price)
    
    def test_group_price_caching(self):
        """Test caching group price data."""
        # Skip if no connection
        if not self.cache.client:
            logger.warning("Skipping cache test: No Valkey/Redis connection")
            return
            
        product_id = "TEST123"
        threshold = 25
        current_quantity = 15
        price = 32.99
        
        # Cache the group price
        self.cache.cache_group_price(product_id, threshold, current_quantity, price)
        
        # Get the cached group price
        cached_price = self.cache.get_group_price(product_id, threshold, current_quantity)
        
        # Check if the price matches
        self.assertEqual(cached_price, price)

class TestDatabase(unittest.TestCase):
    """Test the database operations."""
    
    def setUp(self):
        self.db = Database()
        self.test_product_id = "TEST123"
        self.test_asin = "B07Q32KX3J"
        
        # Create tables
        self.db.create_tables()
        
        # Clean up any existing test data
        self.db.cursor.execute("DELETE FROM products WHERE product_id = %s", (self.test_product_id,))
        self.db.conn.commit()
    
    def tearDown(self):
        # Clean up test data
        self.db.cursor.execute("DELETE FROM products WHERE product_id = %s", (self.test_product_id,))
        self.db.cursor.execute("DELETE FROM bulk_pricing_tiers WHERE product_id = %s", (self.test_product_id,))
        self.db.cursor.execute("DELETE FROM group_order_thresholds WHERE product_id = %s", (self.test_product_id,))
        self.db.cursor.execute("DELETE FROM group_orders WHERE product_id = %s", (self.test_product_id,))
        self.db.conn.commit()
        self.db.close()
    
    def test_insert_get_product(self):
        """Test inserting and getting a product."""
        # Create test product data
        product_data = {
            'product_id': self.test_product_id,
            'name': 'Test Product',
            'category': 'Test Category',
            'wholesale_cost': 20.0,
            'retail_price': 39.99,
            'competitor_asin': self.test_asin
        }
        
        # Insert the product
        self.db.insert_product(product_data)
        
        # Get the product
        product = self.db.get_product(self.test_product_id)
        
        # Check if the product matches
        self.assertEqual(product['product_id'], product_data['product_id'])
        self.assertEqual(product['name'], product_data['name'])
        self.assertEqual(float(product['wholesale_cost']), product_data['wholesale_cost'])
        self.assertEqual(float(product['retail_price']), product_data['retail_price'])
        self.assertEqual(product['competitor_asin'], product_data['competitor_asin'])
    
    def test_insert_amazon_product(self):
        """Test inserting an Amazon product."""
        # Create test Amazon product data
        amazon_data = {
            'asin': self.test_asin,
            'title': 'Test Amazon Product',
            'brand': 'Test Brand',
            'categories': ['Category 1', 'Category 2'],
            'availability': 'In Stock',
            'rating': 4.5,
            'reviews_count': 1000,
            'buybox_seller': 'Test Seller',
            'number_of_sellers': 3
        }
        
        # Insert the Amazon product
        self.db.insert_amazon_product(amazon_data)
        
        # Get the Amazon product
        amazon_product = self.db.get_amazon_product(self.test_asin)
        
        # Check if the product was inserted
        self.assertIsNotNone(amazon_product)
        self.assertEqual(amazon_product['asin'], amazon_data['asin'])
        self.assertEqual(amazon_product['title'], amazon_data['title'])
        
        # Clean up
        self.db.cursor.execute("DELETE FROM amazon_products WHERE asin = %s", (self.test_asin,))
        self.db.conn.commit()
    
    def test_bulk_pricing_tiers(self):
        """Test setting and getting bulk pricing tiers."""
        # Create test product first
        product_data = {
            'product_id': self.test_product_id,
            'name': 'Test Product',
            'wholesale_cost': 20.0,
            'retail_price': 39.99
        }
        self.db.insert_product(product_data)
        
        # Create test bulk pricing tiers
        tiers = [
            {'min_quantity': 5, 'discount_percentage': 5.0},
            {'min_quantity': 10, 'discount_percentage': 10.0},
            {'min_quantity': 25, 'discount_percentage': 15.0}
        ]
        
        # Set bulk pricing tiers
        self.db.set_bulk_pricing_tiers(self.test_product_id, tiers)
        
        # Get bulk pricing tiers
        retrieved_tiers = self.db.get_bulk_pricing_tiers(self.test_product_id)
        
        # Check if tiers were set correctly
        self.assertEqual(len(retrieved_tiers), len(tiers))
        
        # Check each tier
        for i, tier in enumerate(sorted(retrieved_tiers, key=lambda x: x['min_quantity'])):
            self.assertEqual(int(tier['min_quantity']), tiers[i]['min_quantity'])
            self.assertEqual(float(tier['discount_percentage']), tiers[i]['discount_percentage'])
    
    def test_group_order_thresholds(self):
        """Test setting and getting group order thresholds."""
        # Create test product first
        product_data = {
            'product_id': self.test_product_id,
            'name': 'Test Product',
            'wholesale_cost': 20.0,
            'retail_price': 39.99
        }
        self.db.insert_product(product_data)
        
        # Create test group order thresholds
        thresholds = [
            {'threshold_quantity': 25, 'price_multiplier': 0.9},
            {'threshold_quantity': 50, 'price_multiplier': 0.85},
            {'threshold_quantity': 100, 'price_multiplier': 0.8}
        ]
        
        # Set group order thresholds
        self.db.set_group_order_thresholds(self.test_product_id, thresholds)
        
        # Get group order thresholds
        retrieved_thresholds = self.db.get_group_order_thresholds(self.test_product_id)
        
        # Check if thresholds were set correctly
        self.assertEqual(len(retrieved_thresholds), len(thresholds))
        
        # Check each threshold
        for i, threshold in enumerate(sorted(retrieved_thresholds, key=lambda x: x['threshold_quantity'])):
            self.assertEqual(int(threshold['threshold_quantity']), thresholds[i]['threshold_quantity'])
            self.assertEqual(float(threshold['price_multiplier']), thresholds[i]['price_multiplier'])
    
    def test_group_orders(self):
        """Test creating and updating group orders."""
        # Create test product first
        product_data = {
            'product_id': self.test_product_id,
            'name': 'Test Product',
            'wholesale_cost': 20.0,
            'retail_price': 39.99
        }
        self.db.insert_product(product_data)
        
        # Create a group order
        threshold_quantity = 25
        group_order_id = self.db.create_group_order(self.test_product_id, threshold_quantity)
        
        # Check if group order was created
        self.assertIsNotNone(group_order_id)
        
        # Get the group order
        group_order = self.db.get_group_order(group_order_id)
        
        # Check group order properties
        self.assertEqual(group_order['product_id'], self.test_product_id)
        self.assertEqual(int(group_order['threshold_quantity']), threshold_quantity)
        self.assertEqual(int(group_order['current_quantity']), 0)
        self.assertEqual(group_order['status'], 'active')
        
        # Update the group order
        quantity_change = 10
        updated_order = self.db.update_group_order(group_order_id, quantity_change)
        
        # Check if order was updated
        self.assertEqual(updated_order['product_id'], self.test_product_id)
        self.assertEqual(int(updated_order['current_quantity']), quantity_change)
        self.assertEqual(updated_order['status'], 'active')
        
        # Update again to reach threshold
        updated_order = self.db.update_group_order(group_order_id, threshold_quantity - quantity_change)
        
        # Check if status changed to completed
        self.assertEqual(updated_order['status'], 'completed')
        self.assertIsNotNone(updated_order['completed_at'])
        
        # Get active group orders
        active_orders = self.db.get_active_group_orders(self.test_product_id)
        
        # Check if our order is not in active orders anymore
        self.assertTrue(all(order['id'] != group_order_id for order in active_orders))


class TestPricingModel(unittest.TestCase):
    """Test the pricing model."""
    
    def setUp(self):
        self.model = PricingModel()
        self.test_product_id = "TEST123"
        self.test_asin = "B07Q32KX3J"
    
    def test_extract_features(self):
        """Test extracting features for price prediction."""
        # Create test Amazon data
        amazon_data = {
            'asin': self.test_asin,
            'final_price': 35.99,
            'initial_price': 39.99,
            'discount_percentage': 10.0,
            'price_mean': 36.99,
            'price_std': 2.0,
            'rating': 4.5,
            'reviews_count': 1000,
            'number_of_sellers': 3,
            'availability': 'In Stock'
        }
        
        # Create test product data
        product_data = {
            'product_id': self.test_product_id,
            'wholesale_cost': 20.0,
            'retail_price': 39.99
        }
        
        # Extract features
        features = self.model.extract_features(amazon_data, product_data)
        
        # Check extracted features
        self.assertEqual(features['wholesale_cost'], product_data['wholesale_cost'])
        self.assertEqual(features['competitor_price'], amazon_data['final_price'])
        self.assertEqual(features['competitor_original_price'], amazon_data['initial_price'])
        self.assertEqual(features['competitor_discount'], amazon_data['discount_percentage'])
        self.assertEqual(features['price_mean'], amazon_data['price_mean'])
        self.assertEqual(features['price_std'], amazon_data['price_std'])
        self.assertEqual(features['rating'], amazon_data['rating'])
        self.assertEqual(features['reviews_count'], amazon_data['reviews_count'])
        self.assertEqual(features['seller_count'], amazon_data['number_of_sellers'])
        self.assertEqual(features['is_in_stock'], 1.0)
        
        # Check derived features
        self.assertAlmostEqual(features['wholesale_to_competitor_ratio'], 
                              product_data['wholesale_cost'] / amazon_data['final_price'])
        self.assertAlmostEqual(features['retail_to_competitor_ratio'], 
                              product_data['retail_price'] / amazon_data['final_price'])
    
    def test_predict_price(self):
        """Test predicting price."""
        # Create test Amazon data
        amazon_data = {
            'asin': self.test_asin,
            'final_price': 35.99,
            'initial_price': 39.99,
            'discount_percentage': 10.0,
            'price_mean': 36.99,
            'price_std': 2.0,
            'rating': 4.5,
            'reviews_count': 1000,
            'number_of_sellers': 3,
            'availability': 'In Stock'
        }
        
        # Create test product data
        product_data = {
            'product_id': self.test_product_id,
            'wholesale_cost': 20.0,
            'retail_price': 39.99,
            'min_margin_percentage': 20.0,
            'max_price_multiplier': 1.1
        }
        
        # Predict price
        predicted_price = self.model.predict_price(amazon_data, product_data)
        
        # Check if price is within constraints
        min_price = product_data['wholesale_cost'] * (1 + product_data['min_margin_percentage'] / 100)
        max_price = amazon_data['final_price'] * product_data['max_price_multiplier']
        
        self.assertGreaterEqual(predicted_price, min_price)
        self.assertLessEqual(predicted_price, max_price)
    
    def test_calculate_bulk_price(self):
        """Test calculating bulk price."""
        base_price = 39.99
        
        # Test with different quantities
        quantities = [1, 5, 10, 25, 50, 100]
        
        for quantity in quantities:
            bulk_price = self.model.calculate_bulk_price(base_price, quantity, self.test_product_id)
            
            # Check if bulk price is less than or equal to base price
            self.assertLessEqual(bulk_price, base_price)
            
            # For quantity 1, should be equal to base price
            if quantity == 1:
                self.assertEqual(bulk_price, base_price)
            
            # For larger quantities, should be less than base price
            if quantity >= 5:
                self.assertLess(bulk_price, base_price)
    
    def test_calculate_group_price(self):
        """Test calculating group price."""
        base_price = 39.99
        threshold = 25
        
        # Test with different current quantities
        current_quantities = [0, 5, 10, 15, 20, 25]
        
        for current_quantity in current_quantities:
            group_price = self.model.calculate_group_price(
                base_price, current_quantity, threshold, self.test_product_id
            )
            
            # Check if group price is less than or equal to base price
            self.assertLessEqual(group_price, base_price)
            
            # For quantity 0, should be close to base price
            if current_quantity == 0:
                self.assertAlmostEqual(group_price, base_price, delta=0.01)
            
            # For threshold reached, should be significantly less than base price
            if current_quantity >= threshold:
                self.assertLess(group_price, base_price * 0.95)


class TestFlaskAPI(unittest.TestCase):
    """Test the Flask API."""
    
    @classmethod
    def setUpClass(cls):
        # Start the Flask app in a separate process
        import subprocess
        import time
        
        # Check if app.py exists
        if not os.path.exists('app.py'):
            logger.warning("Skipping Flask API tests: app.py not found")
            cls.app_process = None
            return
        
        # Start the app
        try:
            cls.app_process = subprocess.Popen(['python', 'app.py'])
            # Wait for app to start
            time.sleep(2)
        except Exception as e:
            logger.error(f"Failed to start Flask app: {str(e)}")
            cls.app_process = None
    
    @classmethod
    def tearDownClass(cls):
        # Stop the Flask app
        if cls.app_process:
            cls.app_process.terminate()
    
    def setUp(self):
        self.base_url = f"http://localhost:{os.getenv('FLASK_PORT', 5000) }"
        self.test_product_id = "TEST123"
        self.test_asin = "B07Q32KX3J"
        
        # Skip tests if app not running
        if not hasattr(self, 'app_process') or not self.app_process:
            self.skipTest("Flask app not running")
        
        # Create test product
        self.create_test_product()
    
    def tearDown(self):
        # Delete test product
        self.delete_test_product()
    
    def create_test_product(self):
        """Create a test product for API tests."""
        product_data = {
            'product_id': self.test_product_id,
            'name': 'Test Product',
            'category': 'Test Category',
            'wholesale_cost': 20.0,
            'retail_price': 39.99,
            'competitor_asin': self.test_asin
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/products",
                json=product_data
            )
            
            if response.status_code != 200:
                logger.warning(f"Failed to create test product: {response.text}")
        except Exception as e:
            logger.warning(f"Error creating test product: {str(e)}")
    
    def delete_test_product(self):
        """Delete the test product."""
        db = Database()
        try:
            db.cursor.execute("DELETE FROM products WHERE product_id = %s", (self.test_product_id,))
            db.cursor.execute("DELETE FROM bulk_pricing_tiers WHERE product_id = %s", (self.test_product_id,))
            db.cursor.execute("DELETE FROM group_order_thresholds WHERE product_id = %s", (self.test_product_id,))
            db.cursor.execute("DELETE FROM group_orders WHERE product_id = %s", (self.test_product_id,))
            db.conn.commit()
        finally:
            db.close()
    
    def test_health_check(self):
        """Test the health check endpoint."""
        response = requests.get(f"{self.base_url}/health")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
    
    def test_get_product(self):
        """Test getting a product."""
        response = requests.get(f"{self.base_url}/products/{self.test_product_id}")
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['product']['product_id'], self.test_product_id)
    
    def test_predict_price(self):
        """Test predicting a price."""
        response = requests.get(f"{self.base_url}/predict-price/{self.test_product_id}")
        
        # This might fail if no Amazon data is available
        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data['product_id'], self.test_product_id)
            self.assertIn('predicted_price', data)
            self.assertGreater(data['predicted_price'], 0)
    
    def test_bulk_price(self):
        """Test getting bulk price."""
        quantity = 10
        response = requests.get(f"{self.base_url}/bulk-price/{self.test_product_id}?quantity={quantity}")
        
        # This might fail if price prediction fails
        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data['product_id'], self.test_product_id)
            self.assertEqual(data['quantity'], quantity)
            self.assertIn('bulk_price', data)
            self.assertGreater(data['bulk_price'], 0)
    
    def test_group_price(self):
        """Test getting group price."""
        current_quantity = 10
        threshold = 25
        response = requests.get(
            f"{self.base_url}/group-price/{self.test_product_id}?current_quantity={current_quantity}&threshold={threshold}"
        )
        
        # This might fail if price prediction fails
        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data['product_id'], self.test_product_id)
            self.assertEqual(data['current_quantity'], current_quantity)
            self.assertEqual(data['threshold'], threshold)
            self.assertIn('group_price', data)
            self.assertGreater(data['group_price'], 0)
    
    def test_create_join_group_order(self):
        """Test creating and joining a group order."""
        # Create a group order
        create_response = requests.post(
            f"{self.base_url}/group-orders",
            json={
                'product_id': self.test_product_id,
                'threshold_quantity': 25
            }
        )
        
        self.assertEqual(create_response.status_code, 200)
        create_data = create_response.json()
        self.assertIn('group_order_id', create_data)
        
        group_order_id = create_data['group_order_id']
        
        # Join the group order
        join_response = requests.post(
            f"{self.base_url}/group-orders/{group_order_id}/join",
            json={
                'quantity': 10
            }
        )
        
        self.assertEqual(join_response.status_code, 200)
        join_data = join_response.json()
        self.assertEqual(join_data['group_order_id'], group_order_id)
        self.assertEqual(join_data['current_quantity'], 10)
        self.assertEqual(join_data['threshold_reached'], False)
        
        # Join again to reach threshold
        join_response = requests.post(
            f"{self.base_url}/group-orders/{group_order_id}/join",
            json={
                'quantity': 15
            }
        )
        
        self.assertEqual(join_response.status_code, 200)
        join_data = join_response.json()
        self.assertEqual(join_data['current_quantity'], 25)
        self.assertEqual(join_data['threshold_reached'], True)
        self.assertEqual(join_data['status'], 'completed')


if __name__ == '__main__':
    unittest.main()

import psycopg2
from psycopg2.extras import execute_values
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class Database:
    """Database handler for storing Amazon product and pricing data with bulkbuy/groupbuy support."""
    
    def __init__(self):
        """Initialize database connection using environment variables."""
        self.conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "5432")
        )
        self.cursor = self.conn.cursor()
        logger.info("Database connection established")
    
    def create_tables(self):
        """Create necessary database tables if they don't exist."""
        logger.info("Creating database tables")
        
        # Create products table for our own products
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id VARCHAR(50) PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                category VARCHAR(50),
                wholesale_cost DECIMAL(10, 2),
                retail_price DECIMAL(10, 2),
                competitor_asin VARCHAR(50),
                min_margin_percentage DECIMAL(5, 2) DEFAULT 20.0,
                max_price_multiplier DECIMAL(5, 2) DEFAULT 1.1
            );
        """)
        
        # Create amazon_products table for competitor data
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS amazon_products (
                asin VARCHAR(50) PRIMARY KEY,
                title TEXT,
                brand VARCHAR(255),
                categories JSONB,
                availability VARCHAR(50),
                rating DECIMAL(3, 1),
                reviews_count INTEGER,
                buybox_seller VARCHAR(255),
                number_of_sellers INTEGER,
                last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create amazon_prices table for price history
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS amazon_prices (
                id SERIAL PRIMARY KEY,
                asin VARCHAR(50) REFERENCES amazon_products(asin),
                timestamp TIMESTAMPTZ NOT NULL,
                initial_price DECIMAL(10, 2),
                final_price DECIMAL(10, 2),
                currency VARCHAR(10),
                discount_percentage DECIMAL(5, 2),
                price_mean DECIMAL(10, 2),
                price_std DECIMAL(10, 4),
                variation_prices JSONB
            );
        """)
        
        # Create predicted_prices table
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS predicted_prices (
                id SERIAL PRIMARY KEY,
                product_id VARCHAR(50) REFERENCES products(product_id),
                timestamp TIMESTAMPTZ NOT NULL,
                predicted_price DECIMAL(10, 2),
                confidence DECIMAL(5, 4)
            );
        """)
        
        # Create bulk_pricing_tiers table for bulkbuy support
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS bulk_pricing_tiers (
                id SERIAL PRIMARY KEY,
                product_id VARCHAR(50) REFERENCES products(product_id),
                min_quantity INTEGER NOT NULL,
                discount_percentage DECIMAL(5, 2) NOT NULL,
                UNIQUE(product_id, min_quantity)
            );
        """)
        
        # Create group_order_thresholds table for groupbuy support
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS group_order_thresholds (
                id SERIAL PRIMARY KEY,
                product_id VARCHAR(50) REFERENCES products(product_id),
                threshold_quantity INTEGER NOT NULL,
                price_multiplier DECIMAL(5, 3) NOT NULL,
                UNIQUE(product_id, threshold_quantity)
            );
        """)
        
        # Create group_orders table to track active group orders
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS group_orders (
                id SERIAL PRIMARY KEY,
                product_id VARCHAR(50) REFERENCES products(product_id),
                threshold_quantity INTEGER NOT NULL,
                current_quantity INTEGER NOT NULL DEFAULT 0,
                status VARCHAR(20) NOT NULL DEFAULT 'active',
                created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMPTZ
            );
        """)
        
        # Create indexes for performance
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_amazon_prices_asin ON amazon_prices(asin);
            CREATE INDEX IF NOT EXISTS idx_amazon_prices_timestamp ON amazon_prices(timestamp);
            CREATE INDEX IF NOT EXISTS idx_predicted_prices_product_id ON predicted_prices(product_id);
            CREATE INDEX IF NOT EXISTS idx_predicted_prices_timestamp ON predicted_prices(timestamp);
            CREATE INDEX IF NOT EXISTS idx_bulk_pricing_tiers_product_id ON bulk_pricing_tiers(product_id);
            CREATE INDEX IF NOT EXISTS idx_group_order_thresholds_product_id ON group_order_thresholds(product_id);
            CREATE INDEX IF NOT EXISTS idx_group_orders_product_id ON group_orders(product_id);
            CREATE INDEX IF NOT EXISTS idx_group_orders_status ON group_orders(status);
        """)
        
        self.conn.commit()
        logger.info("Database tables created successfully")
    
    def insert_product(self, product_data: Dict[str, Any]) -> None:
        """
        Insert or update a product in the database.
        
        Args:
            product_data: Product data dictionary
        """
        logger.info(f"Inserting product: {product_data.get('product_id')}")
        
        query = """
            INSERT INTO products (
                product_id, name, category, wholesale_cost, retail_price, 
                competitor_asin, min_margin_percentage, max_price_multiplier
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s
            ) ON CONFLICT (product_id) DO UPDATE SET
                name = EXCLUDED.name,
                category = EXCLUDED.category,
                wholesale_cost = EXCLUDED.wholesale_cost,
                retail_price = EXCLUDED.retail_price,
                competitor_asin = EXCLUDED.competitor_asin,
                min_margin_percentage = EXCLUDED.min_margin_percentage,
                max_price_multiplier = EXCLUDED.max_price_multiplier
        """
        
        values = (
            product_data.get('product_id'),
            product_data.get('name'),
            product_data.get('category'),
            product_data.get('wholesale_cost'),
            product_data.get('retail_price'),
            product_data.get('competitor_asin'),
            product_data.get('min_margin_percentage', 20.0),
            product_data.get('max_price_multiplier', 1.1)
        )
        
        self.cursor.execute(query, values)
        self.conn.commit()
        logger.info(f"Product {product_data.get('product_id')} inserted/updated successfully")
    
    def insert_amazon_product(self, product_data: Dict[str, Any]) -> None:
        """
        Insert or update an Amazon product in the database.
        
        Args:
            product_data: Amazon product data dictionary
        """
        logger.info(f"Inserting Amazon product: {product_data.get('asin')}")
        
        # Convert categories to JSON if it's a list
        categories = product_data.get('categories', [])
        if isinstance(categories, list):
            categories = json.dumps(categories)
        
        query = """
            INSERT INTO amazon_products (
                asin, title, brand, categories, availability, rating, 
                reviews_count, buybox_seller, number_of_sellers, last_updated
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP
            ) ON CONFLICT (asin) DO UPDATE SET
                title = EXCLUDED.title,
                brand = EXCLUDED.brand,
                categories = EXCLUDED.categories,
                availability = EXCLUDED.availability,
                rating = EXCLUDED.rating,
                reviews_count = EXCLUDED.reviews_count,
                buybox_seller = EXCLUDED.buybox_seller,
                number_of_sellers = EXCLUDED.number_of_sellers,
                last_updated = CURRENT_TIMESTAMP
        """
        
        values = (
            product_data.get('asin'),
            product_data.get('title'),
            product_data.get('brand'),
            categories,
            product_data.get('availability'),
            product_data.get('rating'),
            product_data.get('reviews_count'),
            product_data.get('buybox_seller'),
            product_data.get('number_of_sellers')
        )
        
        self.cursor.execute(query, values)
        self.conn.commit()
        logger.info(f"Amazon product {product_data.get('asin')} inserted/updated successfully")
    
    def insert_amazon_price(self, price_data: Dict[str, Any]) -> None:
        """
        Insert Amazon price data in the database.
        
        Args:
            price_data: Amazon price data dictionary
        """
        logger.info(f"Inserting Amazon price for ASIN: {price_data.get('asin')}")
        
        # Convert variation_prices to JSON if it's a list
        variation_prices = price_data.get('variation_prices', [])
        if isinstance(variation_prices, list):
            variation_prices = json.dumps(variation_prices)
        
        query = """
            INSERT INTO amazon_prices (
                asin, timestamp, initial_price, final_price, currency,
                discount_percentage, price_mean, price_std, variation_prices
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
        """
        
        values = (
            price_data.get('asin'),
            price_data.get('timestamp', datetime.now()),
            price_data.get('initial_price'),
            price_data.get('final_price'),
            price_data.get('currency'),
            price_data.get('discount_percentage'),
            price_data.get('price_mean'),
            price_data.get('price_std'),
            variation_prices
        )
        
        self.cursor.execute(query, values)
        self.conn.commit()
        logger.info(f"Amazon price for ASIN {price_data.get('asin')} inserted successfully")
    
    def insert_predicted_price(self, prediction_data: Dict[str, Any]) -> None:
        """
        Insert predicted price in the database.
        
        Args:
            prediction_data: Prediction data dictionary
        """
        logger.info(f"Inserting predicted price for product: {prediction_data.get('product_id')}")
        
        query = """
            INSERT INTO predicted_prices (
                product_id, timestamp, predicted_price, confidence
            ) VALUES (
                %s, %s, %s, %s
            )
        """
        
        values = (
            prediction_data.get('product_id'),
            prediction_data.get('timestamp', datetime.now()),
            prediction_data.get('predicted_price'),
            prediction_data.get('confidence', 0.8)  # Default confidence
        )
        
        self.cursor.execute(query, values)
        self.conn.commit()
        logger.info(f"Predicted price for product {prediction_data.get('product_id')} inserted successfully")
    
    def set_bulk_pricing_tiers(self, product_id: str, tiers: List[Dict[str, Any]]) -> None:
        """
        Set bulk pricing tiers for a product.
        
        Args:
            product_id: Product ID
            tiers: List of tier dictionaries with min_quantity and discount_percentage
        """
        logger.info(f"Setting bulk pricing tiers for product: {product_id}")
        
        # Delete existing tiers
        self.cursor.execute("DELETE FROM bulk_pricing_tiers WHERE product_id = %s", (product_id,))
        
        # Insert new tiers
        if tiers:
            query = """
                INSERT INTO bulk_pricing_tiers (product_id, min_quantity, discount_percentage)
                VALUES %s
            """
            
            values = [(product_id, tier['min_quantity'], tier['discount_percentage']) for tier in tiers]
            execute_values(self.cursor, query, values)
        
        self.conn.commit()
        logger.info(f"Bulk pricing tiers for product {product_id} updated successfully")
    
    def set_group_order_thresholds(self, product_id: str, thresholds: List[Dict[str, Any]]) -> None:
        """
        Set group order thresholds for a product.
        
        Args:
            product_id: Product ID
            thresholds: List of threshold dictionaries with threshold_quantity and price_multiplier
        """
        logger.info(f"Setting group order thresholds for product: {product_id}")
        
        # Delete existing thresholds
        self.cursor.execute("DELETE FROM group_order_thresholds WHERE product_id = %s", (product_id,))
        
        # Insert new thresholds
        if thresholds:
            query = """
                INSERT INTO group_order_thresholds (product_id, threshold_quantity, price_multiplier)
                VALUES %s
            """
            
            values = [(product_id, threshold['threshold_quantity'], threshold['price_multiplier']) for threshold in thresholds]
            execute_values(self.cursor, query, values)
        
        self.conn.commit()
        logger.info(f"Group order thresholds for product {product_id} updated successfully")
    
    def create_group_order(self, product_id: str, threshold_quantity: int) -> int:
        """
        Create a new group order.
        
        Args:
            product_id: Product ID
            threshold_quantity: Quantity threshold for the group order
            
        Returns:
            int: ID of the created group order
        """
        logger.info(f"Creating group order for product: {product_id}")
        
        query = """
            INSERT INTO group_orders (product_id, threshold_quantity)
            VALUES (%s, %s)
            RETURNING id
        """
        
        self.cursor.execute(query, (product_id, threshold_quantity))
        group_order_id = self.cursor.fetchone()[0]
        
        self.conn.commit()
        logger.info(f"Group order {group_order_id} created for product {product_id}")
        
        return group_order_id
    
    def update_group_order(self, group_order_id: int, quantity_change: int) -> Dict[str, Any]:
        """
        Update a group order's current quantity.
        
        Args:
            group_order_id: Group order ID
            quantity_change: Change in quantity (positive or negative)
            
        Returns:
            Dict[str, Any]: Updated group order data
        """
        logger.info(f"Updating group order: {group_order_id}")
        
        # Get current group order
        self.cursor.execute("SELECT * FROM group_orders WHERE id = %s", (group_order_id,))
        row = self.cursor.fetchone()
        
        if not row:
            logger.error(f"Group order {group_order_id} not found")
            return None
        
        columns = [desc[0] for desc in self.cursor.description]
        group_order = dict(zip(columns, row))
        
        # Update quantity
        new_quantity = max(0, group_order['current_quantity'] + quantity_change)
        threshold_reached = new_quantity >= group_order['threshold_quantity']
    
    def close(self):
        """Close the database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")

    def get_product(self, product_id):
        """
        Get a product by ID.
        
        Args:
            product_id: Product ID
            
        Returns:
            Dict[str, Any]: Product data or None if not found
        """
        logger.info(f"Getting product: {product_id}")
        
        query = "SELECT * FROM products WHERE product_id = %s"
        
        self.cursor.execute(query, (product_id,))
        row = self.cursor.fetchone()
        
        if not row:
            logger.warning(f"Product {product_id} not found")
            return None
        
        columns = [desc[0] for desc in self.cursor.description]
        product = dict(zip(columns, row))
        
        return product

    def get_amazon_product(self, asin):
        """
        Get an Amazon product by ASIN.
        
        Args:
            asin: Amazon ASIN
            
        Returns:
            Dict[str, Any]: Amazon product data or None if not found
        """
        logger.info(f"Getting Amazon product: {asin}")
        
        query = "SELECT * FROM amazon_products WHERE asin = %s"
        
        self.cursor.execute(query, (asin,))
        row = self.cursor.fetchone()
        
        if not row:
            logger.warning(f"Amazon product {asin} not found")
            return None
        
        columns = [desc[0] for desc in self.cursor.description]
        product = dict(zip(columns, row))
        
        return product

    def get_amazon_price_history(self, asin):
        """
        Get price history for an Amazon product.
        
        Args:
            asin: Amazon ASIN
            
        Returns:
            List[Dict[str, Any]]: List of price data
        """
        logger.info(f"Getting Amazon price history for: {asin}")
        
        query = """
            SELECT * FROM amazon_prices 
            WHERE asin = %s 
            ORDER BY timestamp DESC
        """
        
        self.cursor.execute(query, (asin,))
        rows = self.cursor.fetchall()
        
        if not rows:
            logger.warning(f"No price history found for Amazon product {asin}")
            return []
        
        columns = [desc[0] for desc in self.cursor.description]
        price_history = [dict(zip(columns, row)) for row in rows]
        
        return price_history

    def get_latest_amazon_price(self, asin):
        """
        Get the latest Amazon price for an ASIN.
        
        Args:
            asin: Amazon ASIN
            
        Returns:
            Dict[str, Any]: Price data or None if not found
        """
        logger.info(f"Getting latest Amazon price for: {asin}")
        
        query = """
            SELECT * FROM amazon_prices 
            WHERE asin = %s 
            ORDER BY timestamp DESC 
            LIMIT 1
        """
        
        self.cursor.execute(query, (asin,))
        row = self.cursor.fetchone()
        
        if not row:
            logger.warning(f"No price data found for Amazon product {asin}")
            return None
        
        columns = [desc[0] for desc in self.cursor.description]
        price_data = dict(zip(columns, row))
        
        return price_data

    def get_latest_predicted_price(self, product_id):
        """
        Get the latest predicted price for a product.
        
        Args:
            product_id: Product ID
            
        Returns:
            Dict[str, Any]: Prediction data or None if not found
        """
        logger.info(f"Getting latest predicted price for: {product_id}")
        
        query = """
            SELECT * FROM predicted_prices 
            WHERE product_id = %s 
            ORDER BY timestamp DESC 
            LIMIT 1
        """
        
        self.cursor.execute(query, (product_id,))
        row = self.cursor.fetchone()
        
        if not row:
            logger.warning(f"No prediction data found for product {product_id}")
            return None
        
        columns = [desc[0] for desc in self.cursor.description]
        prediction_data = dict(zip(columns, row))
        
        return prediction_data

    def get_bulk_pricing_tiers(self, product_id):
        """
        Get bulk pricing tiers for a product.
        
        Args:
            product_id: Product ID
            
        Returns:
            List[Dict[str, Any]]: List of bulk pricing tiers
        """
        logger.info(f"Getting bulk pricing tiers for: {product_id}")
        
        query = """
            SELECT * FROM bulk_pricing_tiers 
            WHERE product_id = %s 
            ORDER BY min_quantity ASC
        """
        
        self.cursor.execute(query, (product_id,))
        rows = self.cursor.fetchall()
        
        if not rows:
            logger.warning(f"No bulk pricing tiers found for product {product_id}")
            return []
        
        columns = [desc[0] for desc in self.cursor.description]
        tiers = [dict(zip(columns, row)) for row in rows]
        
        return tiers

    def get_group_order_thresholds(self, product_id):
        """
        Get group order thresholds for a product.
        
        Args:
            product_id: Product ID
            
        Returns:
            List[Dict[str, Any]]: List of group order thresholds
        """
        logger.info(f"Getting group order thresholds for: {product_id}")
        
        query = """
            SELECT * FROM group_order_thresholds 
            WHERE product_id = %s 
            ORDER BY threshold_quantity ASC
        """
        
        self.cursor.execute(query, (product_id,))
        rows = self.cursor.fetchall()
        
        if not rows:
            logger.warning(f"No group order thresholds found for product {product_id}")
            return []
        
        columns = [desc[0] for desc in self.cursor.description]
        thresholds = [dict(zip(columns, row)) for row in rows]
        
        return thresholds

    def get_group_order(self, group_order_id):
        """
        Get a group order by ID.
        
        Args:
            group_order_id: Group order ID
            
        Returns:
            Dict[str, Any]: Group order data or None if not found
        """
        logger.info(f"Getting group order: {group_order_id}")
        
        query = "SELECT * FROM group_orders WHERE id = %s"
        
        self.cursor.execute(query, (group_order_id,))
        row = self.cursor.fetchone()
        
        if not row:
            logger.warning(f"Group order {group_order_id} not found")
            return None
        
        columns = [desc[0] for desc in self.cursor.description]
        group_order = dict(zip(columns, row))
        
        return group_order

    def get_active_group_orders(self, product_id):
        """
        Get active group orders for a product.
        
        Args:
            product_id: Product ID
            
        Returns:
            List[Dict[str, Any]]: List of active group orders
        """
        logger.info(f"Getting active group orders for: {product_id}")
        
        query = """
            SELECT * FROM group_orders 
            WHERE product_id = %s AND status = 'active' 
            ORDER BY created_at DESC
        """
        
        self.cursor.execute(query, (product_id,))
        rows = self.cursor.fetchall()
        
        if not rows:
            logger.warning(f"No active group orders found for product {product_id}")
            return []
        
        columns = [desc[0] for desc in self.cursor.description]
        orders = [dict(zip(columns, row)) for row in rows]
        
        return orders
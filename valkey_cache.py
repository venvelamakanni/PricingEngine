import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class ValkeyCache:
    """
    Caching system using Valkey (Redis-compatible) to reduce API calls.
    Provides methods for caching Amazon product data, pricing information,
    and predicted prices with support for bulk and group pricing.
    """
    
    def __init__(self):
        """Initialize Valkey connection using environment variables."""
        self.host = os.getenv("VALKEY_HOST", "localhost")
        self.port = int(os.getenv("VALKEY_PORT", "6379"))
        self.db = int(os.getenv("VALKEY_DB", "0"))
        self.password = os.getenv("VALKEY_PASSWORD", None)
        self.default_ttl = int(os.getenv("VALKEY_DEFAULT_TTL", "86400"))  # 24 hours
        
        # Import here to avoid dependency issues if Redis/Valkey is not installed
        try:
            import redis
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True
            )
            logger.info(f"Connected to Valkey/Redis at {self.host}:{self.port}")
            
            # Test connection
            self.client.ping()
        except ImportError:
            logger.error("Redis package not installed. Install with: pip install redis")
            raise
        except Exception as e:
            logger.error(f"Failed to connect to Valkey/Redis: {str(e)}")
            self.client = None
    
    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """
        Set a string value in the cache.
        
        Args:
            key: Cache key
            value: String value to cache
            ttl: Time-to-live in seconds (None for default TTL)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("No Valkey/Redis connection available")
            return False
        
        try:
            ttl = ttl if ttl is not None else self.default_ttl
            self.client.set(key, value, ex=ttl)
            logger.debug(f"Set cache key: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {str(e)}")
            return False
    
    def get(self, key: str) -> Optional[str]:
        """
        Get a string value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[str]: Cached value or None if not found
        """
        if not self.client:
            logger.error("No Valkey/Redis connection available")
            return None
        
        try:
            value = self.client.get(key)
            if value:
                logger.debug(f"Cache hit for key: {key}")
            else:
                logger.debug(f"Cache miss for key: {key}")
            return value
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {str(e)}")
            return None
    
    def set_json(self, key: str, value: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Set a JSON value in the cache.
        
        Args:
            key: Cache key
            value: Dictionary to cache as JSON
            ttl: Time-to-live in seconds (None for default TTL)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            json_value = json.dumps(value, default=str)
            return self.set(key, json_value, ttl)
        except Exception as e:
            logger.error(f"Error serializing JSON for key {key}: {str(e)}")
            return False
    
    def get_json(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get a JSON value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[Dict[str, Any]]: Deserialized JSON or None if not found
        """
        value = self.get(key)
        if not value:
            return None
        
        try:
            return json.loads(value)
        except Exception as e:
            logger.error(f"Error deserializing JSON for key {key}: {str(e)}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("No Valkey/Redis connection available")
            return False
        
        try:
            self.client.delete(key)
            logger.debug(f"Deleted cache key: {key}")
            return True
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {str(e)}")
            return False
    
    def ttl(self, key: str) -> int:
        """
        Get the remaining TTL for a key.
        
        Args:
            key: Cache key
            
        Returns:
            int: Remaining TTL in seconds, -1 if no TTL, -2 if key doesn't exist
        """
        if not self.client:
            logger.error("No Valkey/Redis connection available")
            return -2
        
        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.error(f"Error getting TTL for key {key}: {str(e)}")
            return -2
    
    def cache_product(self, asin: str, product_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Cache Amazon product data.
        
        Args:
            asin: Amazon Standard Identification Number
            product_data: Product data to cache
            ttl: Time-to-live in seconds (None for default TTL)
            
        Returns:
            bool: True if successful, False otherwise
        """
        key = f"product:{asin}"
        logger.info(f"Caching product data for ASIN: {asin}")
        return self.set_json(key, product_data, ttl)
    
    def get_product(self, asin: str) -> Optional[Dict[str, Any]]:
        """
        Get cached Amazon product data.
        
        Args:
            asin: Amazon Standard Identification Number
            
        Returns:
            Optional[Dict[str, Any]]: Cached product data or None if not found
        """
        key = f"product:{asin}"
        logger.info(f"Getting cached product data for ASIN: {asin}")
        return self.get_json(key)
    
    def cache_pricing(self, asin: str, pricing_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Cache Amazon pricing data.
        
        Args:
            asin: Amazon Standard Identification Number
            pricing_data: Pricing data to cache
            ttl: Time-to-live in seconds (None for default TTL)
            
        Returns:
            bool: True if successful, False otherwise
        """
        key = f"pricing:{asin}"
        logger.info(f"Caching pricing data for ASIN: {asin}")
        return self.set_json(key, pricing_data, ttl)
    
    def get_pricing(self, asin: str) -> Optional[Dict[str, Any]]:
        """
        Get cached Amazon pricing data.
        
        Args:
            asin: Amazon Standard Identification Number
            
        Returns:
            Optional[Dict[str, Any]]: Cached pricing data or None if not found
        """
        key = f"pricing:{asin}"
        logger.info(f"Getting cached pricing data for ASIN: {asin}")
        return self.get_json(key)
    
    def cache_predicted_price(self, product_id: str, price: float, ttl: Optional[int] = None) -> bool:
        """
        Cache predicted price for a product.
        
        Args:
            product_id: Product ID
            price: Predicted price
            ttl: Time-to-live in seconds (None for default TTL)
            
        Returns:
            bool: True if successful, False otherwise
        """
        key = f"predicted_price:{product_id}"
        logger.info(f"Caching predicted price for product: {product_id}")
        return self.set(key, str(price), ttl)
    
    def get_predicted_price(self, product_id: str) -> Optional[float]:
        """
        Get cached predicted price for a product.
        
        Args:
            product_id: Product ID
            
        Returns:
            Optional[float]: Cached predicted price or None if not found
        """
        key = f"predicted_price:{product_id}"
        logger.info(f"Getting cached predicted price for product: {product_id}")
        value = self.get(key)
        
        if value is not None:
            try:
                return float(value)
            except ValueError:
                logger.error(f"Invalid cached price value for product {product_id}: {value}")
        
        return None
    
    def cache_bulk_price(self, product_id: str, quantity: int, price: float, ttl: Optional[int] = None) -> bool:
        """
        Cache bulk price for a product and quantity.
        
        Args:
            product_id: Product ID
            quantity: Order quantity
            price: Bulk price
            ttl: Time-to-live in seconds (None for default TTL)
            
        Returns:
            bool: True if successful, False otherwise
        """
        key = f"bulk_price:{product_id}:{quantity}"
        logger.info(f"Caching bulk price for product: {product_id}, quantity: {quantity}")
        return self.set(key, str(price), ttl)
    
    def get_bulk_price(self, product_id: str, quantity: int) -> Optional[float]:
        """
        Get cached bulk price for a product and quantity.
        
        Args:
            product_id: Product ID
            quantity: Order quantity
            
        Returns:
            Optional[float]: Cached bulk price or None if not found
        """
        key = f"bulk_price:{product_id}:{quantity}"
        logger.info(f"Getting cached bulk price for product: {product_id}, quantity: {quantity}")
        value = self.get(key)
        
        if value is not None:
            try:
                return float(value)
            except ValueError:
                logger.error(f"Invalid cached bulk price value for product {product_id}: {value}")
        
        return None
    
    def cache_group_price(self, product_id: str, threshold: int, current_quantity: int, price: float, ttl: Optional[int] = None) -> bool:
        """
        Cache group price for a product.
        
        Args:
            product_id: Product ID
            threshold: Group order threshold
            current_quantity: Current quantity in the group order
            price: Group price
            ttl: Time-to-live in seconds (None for default TTL)
            
        Returns:
            bool: True if successful, False otherwise
        """
        key = f"group_price:{product_id}:{threshold}:{current_quantity}"
        logger.info(f"Caching group price for product: {product_id}, threshold: {threshold}, current: {current_quantity}")
        return self.set(key, str(price), ttl)
    
    def get_group_price(self, product_id: str, threshold: int, current_quantity: int) -> Optional[float]:
        """
        Get cached group price for a product.
        
        Args:
            product_id: Product ID
            threshold: Group order threshold
            current_quantity: Current quantity in the group order
            
        Returns:
            Optional[float]: Cached group price or None if not found
        """
        key = f"group_price:{product_id}:{threshold}:{current_quantity}"
        logger.info(f"Getting cached group price for product: {product_id}, threshold: {threshold}, current: {current_quantity}")
        value = self.get(key)
        
        if value is not None:
            try:
                return float(value)
            except ValueError:
                logger.error(f"Invalid cached group price value for product {product_id}: {value}")
        
        return None
    
    def cache_market_analysis(self, asin: str, analysis_data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """
        Cache market analysis for a product.
        
        Args:
            asin: Amazon Standard Identification Number
            analysis_data: Market analysis data
            ttl: Time-to-live in seconds (None for default TTL)
            
        Returns:
            bool: True if successful, False otherwise
        """
        key = f"market_analysis:{asin}"
        logger.info(f"Caching market analysis for ASIN: {asin}")
        return self.set_json(key, analysis_data, ttl)
    
    def get_market_analysis(self, asin: str) -> Optional[Dict[str, Any]]:
        """
        Get cached market analysis for a product.
        
        Args:
            asin: Amazon Standard Identification Number
            
        Returns:
            Optional[Dict[str, Any]]: Cached market analysis or None if not found
        """
        key = f"market_analysis:{asin}"
        logger.info(f"Getting cached market analysis for ASIN: {asin}")
        return self.get_json(key)
    
    def track_api_calls(self, reset: bool = False) -> int:
        """
        Track and count API calls to monitor usage.
        
        Args:
            reset: Whether to reset the counter
            
        Returns:
            int: Current API call count
        """
        key = "api_call_count"
        
        if reset:
            self.delete(key)
            return 0
        
        if not self.client:
            logger.error("No Valkey/Redis connection available")
            return 0
        
        try:
            count = self.client.incr(key)
            logger.debug(f"API call count: {count}")
            return count
        except Exception as e:
            logger.error(f"Error tracking API calls: {str(e)}")
            return 0
    
    def flush_cache(self) -> bool:
        """
        Flush the entire cache (use with caution).
        
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.client:
            logger.error("No Valkey/Redis connection available")
            return False
        
        try:
            self.client.flushdb()
            logger.info("Cache flushed successfully")
            return True
        except Exception as e:
            logger.error(f"Error flushing cache: {str(e)}")
            return False

# Example usage
if __name__ == "__main__":
    cache = ValkeyCache()
    cache.set("test_key", "test_value")
    print(f"Test value: {cache.get('test_key')}")
    cache.delete("test_key")

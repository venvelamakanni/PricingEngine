import requests
import logging
import os
import json
import time
import statistics
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class AmazonAPI:
    """Client for fetching competitor pricing data from Amazon."""
    
    def __init__(self):
        """Initialize the Amazon API client with credentials from environment variables."""
        self.api_key = os.getenv("AMAZON_API_KEY")
        self.api_secret = os.getenv("AMAZON_API_SECRET")
        self.base_url = os.getenv("AMAZON_API_URL", "https://api.rainforestapi.com/request")
        self.max_retries = int(os.getenv("AMAZON_MAX_RETRIES", "3"))
        self.retry_delay = int(os.getenv("AMAZON_RETRY_DELAY", "2"))
        
        # Validate required credentials
        if not self.api_key:
            logger.warning("Amazon API key not configured. Set AMAZON_API_KEY in .env file.")
    
    def get_product(self, asin: str) -> Dict[str, Any]:
        """
        Get detailed product information by ASIN.
        
        Args:
            asin: Amazon Standard Identification Number
            
        Returns:
            Dict[str, Any]: Product information
        """
        logger.info(f"Fetching product data for ASIN: {asin}")
        
        params = {
            'api_key': self.api_key,
            'type': 'product',
            'amazon_domain': 'amazon.com',
            'asin': asin
        }
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Successfully fetched data for ASIN: {asin}")
                    return data.get('product', {})
                elif response.status_code == 429:
                    logger.warning(f"Rate limit exceeded. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        return {"error": f"Failed after {self.max_retries} attempts: {response.text}"}
                    
            except Exception as e:
                logger.error(f"Error fetching product data: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return {"error": f"Exception: {str(e)}"}
        
        return {"error": "Maximum retries exceeded"}
    
    def search_products(self, query: str, category: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for products using keywords.
        
        Args:
            query: Search query
            category: Optional category to filter results
            limit: Maximum number of results to return
            
        Returns:
            List[Dict[str, Any]]: List of product information
        """
        logger.info(f"Searching products with query: {query}")
        
        params = {
            'api_key': self.api_key,
            'type': 'search',
            'amazon_domain': 'amazon.com',
            'search_term': query,
            'sort_by': 'featured'
        }
        
        if category:
            params['category_id'] = category
        
        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.base_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"Successfully searched products for query: {query}")
                    search_results = data.get('search_results', [])
                    return search_results[:limit]
                elif response.status_code == 429:
                    logger.warning(f"Rate limit exceeded. Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"API request failed with status {response.status_code}: {response.text}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                    else:
                        return [{"error": f"Failed after {self.max_retries} attempts: {response.text}"}]
                    
            except Exception as e:
                logger.error(f"Error searching products: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    return [{"error": f"Exception: {str(e)}"}]
        
        return [{"error": "Maximum retries exceeded"}]
    
    def get_competitor_prices(self, asins: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get competitor pricing information for multiple products.
        
        Args:
            asins: List of ASINs to fetch pricing for
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary mapping ASINs to pricing information
        """
        logger.info(f"Fetching competitor prices for {len(asins)} ASINs")
        
        results = {}
        
        for asin in asins:
            try:
                # Add a small delay between requests to avoid rate limiting
                if asin != asins[0]:
                    time.sleep(0.5)
                
                product_data = self.get_product(asin)
                results[asin] = product_data
                
            except Exception as e:
                logger.error(f"Error fetching pricing for ASIN {asin}: {str(e)}")
                results[asin] = {"error": str(e)}
        
        return results
    
    def extract_pricing_data(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract relevant pricing data from Amazon product data.
        
        Args:
            product_data: Raw Amazon product data
            
        Returns:
            Dict[str, Any]: Extracted pricing information
        """
        pricing_data = {
            'asin': product_data.get('asin', ''),
            'title': product_data.get('title', ''),
            'brand': product_data.get('brand', ''),
            'categories': product_data.get('categories', []),
            'initial_price': product_data.get('initial_price', 0),
            'final_price': product_data.get('final_price', 0),
            'currency': product_data.get('currency', 'USD'),
            'availability': product_data.get('availability', ''),
            'rating': product_data.get('rating', 0),
            'reviews_count': product_data.get('reviews_count', 0),
            'buybox_seller': product_data.get('buybox_seller', ''),
            'number_of_sellers': product_data.get('number_of_sellers', 0)
        }
        
        # Extract variation prices if available
        variations = product_data.get('variations', [])
        variation_prices = []
        
        if variations:
            for variation in variations:
                if 'price' in variation:
                    variation_prices.append(variation.get('price', 0))
        
        # Calculate price statistics
        if variation_prices:
            pricing_data['variation_prices'] = variation_prices
            pricing_data['price_mean'] = statistics.mean(variation_prices)
            pricing_data['price_std'] = statistics.stdev(variation_prices) if len(variation_prices) > 1 else 0
        else:
            pricing_data['price_mean'] = pricing_data['final_price']
            pricing_data['price_std'] = 0
        
        # Calculate discount percentage
        initial_price = pricing_data['initial_price']
        final_price = pricing_data['final_price']
        
        if initial_price and final_price and initial_price > final_price:
            pricing_data['discount_percentage'] = ((initial_price - final_price) / initial_price) * 100
        else:
            pricing_data['discount_percentage'] = 0
        
        # Add timestamp
        pricing_data['timestamp'] = datetime.now()
        
        return pricing_data
    
    def analyze_market_position(self, product_data: Dict[str, Any], similar_products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze market position compared to similar products.
        
        Args:
            product_data: Target product data
            similar_products: List of similar product data
            
        Returns:
            Dict[str, Any]: Market position analysis
        """
        if not similar_products:
            return {
                'market_position': 'unknown',
                'price_percentile': 0,
                'rating_percentile': 0,
                'competitive_index': 0
            }
        
        # Extract prices and ratings
        target_price = product_data.get('final_price', 0)
        target_rating = product_data.get('rating', 0)
        
        prices = [p.get('final_price', 0) for p in similar_products if p.get('final_price', 0) > 0]
        ratings = [p.get('rating', 0) for p in similar_products if p.get('rating', 0) > 0]
        
        if not prices or not ratings:
            return {
                'market_position': 'unknown',
                'price_percentile': 0,
                'rating_percentile': 0,
                'competitive_index': 0
            }
        
        # Calculate percentiles
        prices.sort()
        ratings.sort()
        
        price_percentile = sum(1 for p in prices if p >= target_price) / len(prices) * 100
        rating_percentile = sum(1 for r in ratings if r <= target_rating) / len(ratings) * 100
        
        # Calculate competitive index (higher is better)
        # High rating and low price is ideal
        competitive_index = (rating_percentile + (100 - price_percentile)) / 2
        
        # Determine market position
        if competitive_index > 75:
            market_position = 'excellent'
        elif competitive_index > 60:
            market_position = 'good'
        elif competitive_index > 40:
            market_position = 'average'
        elif competitive_index > 25:
            market_position = 'below_average'
        else:
            market_position = 'poor'
        
        return {
            'market_position': market_position,
            'price_percentile': price_percentile,
            'rating_percentile': rating_percentile,
            'competitive_index': competitive_index,
            'avg_competitor_price': statistics.mean(prices) if prices else 0,
            'avg_competitor_rating': statistics.mean(ratings) if ratings else 0
        }
    
    def get_price_recommendations(self, product_data: Dict[str, Any], 
                                 similar_products: List[Dict[str, Any]],
                                 wholesale_cost: float) -> Dict[str, Any]:
        """
        Get price recommendations based on market analysis.
        
        Args:
            product_data: Target product data
            similar_products: List of similar product data
            wholesale_cost: Wholesale cost of the product
            
        Returns:
            Dict[str, Any]: Price recommendations
        """
        market_analysis = self.analyze_market_position(product_data, similar_products)
        
        # Calculate minimum viable price (ensure profit margin)
        min_margin_percentage = 20  # Minimum 20% margin
        min_price = wholesale_cost * (1 + min_margin_percentage / 100)
        
        # Get average and median competitor prices
        prices = [p.get('final_price', 0) for p in similar_products if p.get('final_price', 0) > 0]
        
        if not prices:
            avg_price = product_data.get('final_price', 0) or min_price * 1.5
            median_price = avg_price
        else:
            avg_price = statistics.mean(prices)
            median_price = statistics.median(prices)
        
        # Calculate recommended prices for different strategies
        competitive_price = max(min_price, avg_price * 0.95)  # 5% below average
        optimal_price = max(min_price, median_price * 0.98)  # 2% below median
        premium_price = max(min_price, avg_price * 1.05)  # 5% above average
        
        # Calculate bulk pricing tiers
        bulk_pricing = [
            {'min_quantity': 5, 'price': max(min_price, optimal_price * 0.95)},  # 5% off for 5+ units
            {'min_quantity': 10, 'price': max(min_price, optimal_price * 0.90)},  # 10% off for 10+ units
            {'min_quantity': 25, 'price': max(min_price, optimal_price * 0.85)},  # 15% off for 25+ units
            {'min_quantity': 50, 'price': max(min_price, optimal_price * 0.80)},  # 20% off for 50+ units
            {'min_quantity': 100, 'price': max(min_price, optimal_price * 0.75)}   # 25% off for 100+ units
        ]
        
        # Calculate group buy thresholds
        group_buy = [
            {'threshold': 25, 'price': max(min_price, optimal_price * 0.90)},  # 10% off at 25 units
            {'threshold': 50, 'price': max(min_price, optimal_price * 0.85)},  # 15% off at 50 units
            {'threshold': 100, 'price': max(min_price, optimal_price * 0.80)}   # 20% off at 100 units
        ]
        
        return {
            'market_analysis': market_analysis,
            'min_viable_price': min_price,
            'competitive_price': competitive_price,
            'optimal_price': optimal_price,
            'premium_price': premium_price,
            'bulk_pricing': bulk_pricing,
            'group_buy': group_buy
        }

# Example usage
if __name__ == "__main__":
    api = AmazonAPI()
    product = api.get_product("B07Q32KX3J")
    pricing = api.extract_pricing_data(product)
    print(json.dumps(pricing, indent=2, default=str))

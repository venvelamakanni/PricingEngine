import os
import logging
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

class PricingModel:
    """
    Pricing model that uses Amazon competitor data to predict optimal prices
    with support for bulkbuy and groupbuy business cases.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the pricing model.
        
        Args:
            model_path: Path to saved model file (optional)
        """
        self.model = None
        self.scaler = None
        self.feature_names = [
            'wholesale_cost', 'competitor_price', 'competitor_original_price',
            'competitor_discount', 'price_mean', 'price_std', 'rating',
            'reviews_count', 'seller_count', 'is_in_stock',
            'wholesale_to_competitor_ratio', 'retail_to_competitor_ratio'
        ]
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            try:
                self._load_model(model_path)
                logger.info(f"Loaded pricing model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model from {model_path}: {str(e)}")
                self.model = None
                self.scaler = None
    
    def extract_features(self, amazon_data: Dict[str, Any], product_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract features for price prediction from Amazon and product data.
        
        Args:
            amazon_data: Amazon competitor data
            product_data: Our product data
            
        Returns:
            Dict[str, float]: Features for price prediction
        """
        # Extract basic features
        wholesale_cost = float(product_data.get('wholesale_cost', 0))
        retail_price = float(product_data.get('retail_price', 0))
        competitor_price = float(amazon_data.get('final_price', 0))
        competitor_original_price = float(amazon_data.get('initial_price', 0))
        
        # Handle missing or zero values
        if competitor_price <= 0:
            competitor_price = retail_price
        
        if competitor_original_price <= 0:
            competitor_original_price = competitor_price
        
        # Calculate derived features
        competitor_discount = float(amazon_data.get('discount_percentage', 0))
        price_mean = float(amazon_data.get('price_mean', competitor_price))
        price_std = float(amazon_data.get('price_std', 0))
        
        # Calculate ratios
        wholesale_to_competitor_ratio = wholesale_cost / competitor_price if competitor_price > 0 else 0
        retail_to_competitor_ratio = retail_price / competitor_price if competitor_price > 0 else 1
        
        # Extract other Amazon features
        rating = float(amazon_data.get('rating', 0))
        reviews_count = float(amazon_data.get('reviews_count', 0))
        seller_count = float(amazon_data.get('number_of_sellers', 0))
        is_in_stock = 1.0 if amazon_data.get('availability', '').lower() == 'in stock' else 0.0
        
        # Compile features
        features = {
            'wholesale_cost': wholesale_cost,
            'retail_price': retail_price,
            'competitor_price': competitor_price,
            'competitor_original_price': competitor_original_price,
            'competitor_discount': competitor_discount,
            'price_mean': price_mean,
            'price_std': price_std,
            'rating': rating,
            'reviews_count': reviews_count,
            'seller_count': seller_count,
            'is_in_stock': is_in_stock,
            'wholesale_to_competitor_ratio': wholesale_to_competitor_ratio,
            'retail_to_competitor_ratio': retail_to_competitor_ratio
        }
        
        return features
    
    def predict_price(self, amazon_data: Dict[str, Any], product_data: Dict[str, Any]) -> float:
        """
        Predict optimal price based on Amazon competitor data and product data.
        
        Args:
            amazon_data: Amazon competitor data
            product_data: Our product data
            
        Returns:
            float: Predicted optimal price
        """
        # Extract features
        features = self.extract_features(amazon_data, product_data)
        
        # Get constraints
        wholesale_cost = features['wholesale_cost']
        competitor_price = features['competitor_price']
        min_margin_percentage = float(product_data.get('min_margin_percentage', 20.0))
        max_price_multiplier = float(product_data.get('max_price_multiplier', 1.1))
        
        # Calculate price constraints
        min_price = wholesale_cost * (1 + min_margin_percentage / 100)
        max_price = competitor_price * max_price_multiplier
        
        # Use ML model if available
        if self.model and self.scaler:
            try:
                # Prepare features for model
                feature_vector = np.array([[features[name] for name in self.feature_names]])
                
                # Scale features
                scaled_features = self.scaler.transform(feature_vector)
                
                # Predict price
                predicted_price = self.model.predict(scaled_features)[0]
                
                # Apply constraints
                constrained_price = max(min_price, min(predicted_price, max_price))
                
                logger.info(f"ML model predicted price: ${predicted_price:.2f}, constrained: ${constrained_price:.2f}")
                return constrained_price
            except Exception as e:
                logger.error(f"Error predicting price with ML model: {str(e)}")
                # Fall back to simple pricing strategy
        
        # Use simple pricing strategy if model not available or prediction failed
        price = self._simple_pricing_strategy(features)
        
        # Apply constraints
        constrained_price = max(min_price, min(price, max_price))
        
        logger.info(f"Simple strategy price: ${price:.2f}, constrained: ${constrained_price:.2f}")
        return constrained_price
    
    def calculate_bulk_price(self, base_price: float, quantity: int, product_id: str, 
                            bulk_pricing_tiers: Optional[List[Dict[str, Any]]] = None) -> float:
        """
        Calculate price for bulk orders based on quantity.
        
        Args:
            base_price: Base price for the product
            quantity: Order quantity
            product_id: Product ID
            bulk_pricing_tiers: List of bulk pricing tiers (optional)
            
        Returns:
            float: Bulk price
        """
        # If no tiers provided, use default tiers
        if not bulk_pricing_tiers:
            # Default bulk pricing tiers
            default_tiers = [
                {'min_quantity': 5, 'discount_percentage': 5.0},
                {'min_quantity': 10, 'discount_percentage': 10.0},
                {'min_quantity': 25, 'discount_percentage': 15.0},
                {'min_quantity': 50, 'discount_percentage': 20.0},
                {'min_quantity': 100, 'discount_percentage': 25.0}
            ]
            bulk_pricing_tiers = default_tiers
        
        # Find applicable tier
        applicable_tier = None
        for tier in sorted(bulk_pricing_tiers, key=lambda x: x['min_quantity'], reverse=True):
            if quantity >= tier['min_quantity']:
                applicable_tier = tier
                break
        
        # Apply discount if tier found
        if applicable_tier:
            discount = applicable_tier['discount_percentage'] / 100.0
            bulk_price = base_price * (1 - discount)
            logger.info(f"Applied bulk discount of {applicable_tier['discount_percentage']}% for quantity {quantity}")
            return bulk_price
        
        # No applicable tier found
        return base_price
    
    def calculate_group_price(self, base_price: float, current_quantity: int, threshold: int, product_id: str,
                             group_order_thresholds: Optional[List[Dict[str, Any]]] = None) -> float:
        """
        Calculate price for group orders based on current quantity and threshold.
        
        Args:
            base_price: Base price for the product
            current_quantity: Current quantity in the group order
            threshold: Quantity threshold for the group order
            product_id: Product ID
            group_order_thresholds: List of group order thresholds (optional)
            
        Returns:
            float: Group price
        """
        # If no thresholds provided, use default thresholds
        if not group_order_thresholds:
            # Default group order thresholds
            default_thresholds = [
                {'threshold_quantity': 25, 'price_multiplier': 0.9},
                {'threshold_quantity': 50, 'price_multiplier': 0.85},
                {'threshold_quantity': 100, 'price_multiplier': 0.8}
            ]
            group_order_thresholds = default_thresholds
        
        # Find applicable threshold
        applicable_threshold = None
        for threshold_data in sorted(group_order_thresholds, key=lambda x: x['threshold_quantity'], reverse=True):
            if threshold >= threshold_data['threshold_quantity']:
                applicable_threshold = threshold_data
                break
        
        # Apply multiplier if threshold found
        if applicable_threshold:
            # Calculate progress toward threshold
            progress_ratio = min(current_quantity / threshold, 1.0)
            
            # Apply multiplier based on progress
            multiplier = applicable_threshold['price_multiplier']
            target_price = base_price * multiplier
            
            # Interpolate between base price and target price based on progress
            if progress_ratio >= 1.0:
                # Threshold reached, apply full discount
                group_price = target_price
            else:
                # Partial progress, interpolate
                discount = (base_price - target_price) * progress_ratio
                group_price = base_price - discount
            
            logger.info(f"Group price: ${group_price:.2f} at {progress_ratio:.1%} progress toward threshold {threshold}")
            return group_price
        
        # No applicable threshold found
        return base_price
    
    def _simple_pricing_strategy(self, features: Dict[str, float]) -> float:
        """
        Simple pricing strategy based on competitor price and wholesale cost.
        
        Args:
            features: Extracted features
            
        Returns:
            float: Calculated price
        """
        wholesale_cost = features['wholesale_cost']
        competitor_price = features['competitor_price']
        rating = features['rating']
        
        # Ensure minimum margin
        min_margin = 1.2  # 20% margin
        min_price = wholesale_cost * min_margin
        
        # Base price calculation
        if rating >= 4.5:
            # High rating, price close to competitor
            price = competitor_price * 0.98
        elif rating >= 4.0:
            # Good rating, price slightly below competitor
            price = competitor_price * 0.95
        elif rating >= 3.5:
            # Average rating, price well below competitor
            price = competitor_price * 0.9
        else:
            # Poor rating, price significantly below competitor
            price = competitor_price * 0.85
        
        # Ensure minimum margin
        price = max(price, min_price)
        
        return price
    
    def train_model(self, training_data: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """
        Train the pricing model on historical data using a stacked ensemble approach.
        
        Args:
            training_data: DataFrame with training data
            save_path: Path to save the trained model (optional)
        """
        logger.info("Training pricing model with stacked ensemble")
        
        try:
            # Prepare features and target
            X = training_data[self.feature_names]
            y = training_data['optimal_price']
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Create base models
            xgb_model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            lgbm_model = LGBMRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            catboost_model = CatBoostRegressor(
                iterations=100,
                learning_rate=0.1,
                depth=5,
                subsample=0.8,
                rsm=0.8,
                random_seed=42,
                verbose=False
            )
            
            # Create stacked ensemble
            self.model = StackingRegressor(
                estimators=[
                    ('xgb', xgb_model),
                    ('lgbm', lgbm_model),
                    ('cat', catboost_model)
                ],
                final_estimator=RidgeCV(),
                cv=5
            )
            
            # Train model
            self.model.fit(X_scaled, y)
            
            logger.info("Stacked ensemble model training completed successfully")
            
            # Save model if path provided
            if save_path:
                self._save_model(save_path)
                logger.info(f"Model saved to {save_path}")
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise



    def _save_model(self, path: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            path: Path to save the model
        """
        if self.model and self.scaler:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'timestamp': datetime.now()
            }
            joblib.dump(model_data, path)
    
    def _load_model(self, path: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            path: Path to the saved model
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
    
    def explain_prediction(self, amazon_data: Dict[str, Any], product_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain the factors influencing a price prediction.
        
        Args:
            amazon_data: Amazon competitor data
            product_data: Our product data
            
        Returns:
            Dict[str, Any]: Explanation of the price prediction
        """
        # Extract features
        features = self.extract_features(amazon_data, product_data)
        
        # Get predicted price
        predicted_price = self.predict_price(amazon_data, product_data)
        
        # Calculate key metrics
        wholesale_cost = features['wholesale_cost']
        competitor_price = features['competitor_price']
        
        margin_amount = predicted_price - wholesale_cost
        margin_percentage = (margin_amount / wholesale_cost) * 100 if wholesale_cost > 0 else 0
        
        price_difference = predicted_price - competitor_price
        price_difference_percentage = (price_difference / competitor_price) * 100 if competitor_price > 0 else 0
        
        # Determine key factors
        factors = []
        
        if features['wholesale_to_competitor_ratio'] > 0.8:
            factors.append("High wholesale cost relative to competitor price")
        
        if features['competitor_discount'] > 10:
            factors.append("Competitor is offering a significant discount")
        
        if features['rating'] >= 4.5:
            factors.append("Competitor has excellent ratings")
        elif features['rating'] < 3.5:
            factors.append("Competitor has poor ratings")
        
        if features['reviews_count'] > 1000:
            factors.append("Competitor has many reviews")

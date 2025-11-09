import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd

# =============================================================================
# ASSIGNMENT 1: BASIC FUZZY SET OPERATIONS
# Real-life Example: Restaurant Quality Assessment System
# =============================================================================

class FuzzySet:
    """Class to represent a fuzzy set with elements and their membership values"""
    
    def __init__(self, name: str, elements: Dict[str, float]):
        self.name = name
        self.elements = elements
        
        for key in self.elements:
            self.elements[key] = max(0, min(1, self.elements[key]))
    
    def __str__(self):
        return f"{self.name}: {self.elements}"
    
    def union(self, other):
        """Union operation (maximum of membership values)"""
        union_elements = {}
        all_keys = set(self.elements.keys()) | set(other.elements.keys())
        
        for key in all_keys:
            val1 = self.elements.get(key, 0)
            val2 = other.elements.get(key, 0)
            union_elements[key] = max(val1, val2)
        
        return FuzzySet(f"{self.name} ∪ {other.name}", union_elements)
    
    def intersection(self, other):
        """Intersection operation (minimum of membership values)"""
        intersection_elements = {}
        all_keys = set(self.elements.keys()) | set(other.elements.keys())
        
        for key in all_keys:
            val1 = self.elements.get(key, 0)
            val2 = other.elements.get(key, 0)
            intersection_elements[key] = min(val1, val2)
        
        return FuzzySet(f"{self.name} ∩ {other.name}", intersection_elements)
    
    def complement(self):
        """Complement operation (1 - membership value)"""
        complement_elements = {key: 1 - val for key, val in self.elements.items()}
        return FuzzySet(f"¬{self.name}", complement_elements)
    
    def difference(self, other):
        """Difference operation A - B = A ∩ ¬B"""
        return self.intersection(other.complement())
    

def restaurant_membership_function(rating: float, cuisine_authenticity: float, 
                                 service_quality: float, price_value: float) -> float:
    """
    Membership function for restaurant quality assessment
    All parameters are on a scale of 0-10
    """
    # Weighted combination with normalization
    weights = [0.3, 0.25, 0.25, 0.2]  # Rating, Authenticity, Service, Value
    normalized_values = [rating/10, cuisine_authenticity/10, service_quality/10, price_value/10]
    
    membership = sum(w * v for w, v in zip(weights, normalized_values))
    return min(1.0, max(0.0, membership))




def assignment_1_demo():
    """Assignment 1: Basic Fuzzy Set Operations Demo"""
    print("="*60)
    print("ASSIGNMENT 1: BASIC FUZZY SET OPERATIONS")
    print("Real-life Example: Restaurant Quality Assessment System")
    print("="*60)
    
    # Restaurant data (Rating, Cuisine Authenticity, Service Quality, Price Value)
    restaurants = {
        "Taj mahal": (8.5, 9.0, 7.5, 8.0),
        "Nashik Dabba": (9.0, 8.5, 9.0, 6.0),
        "PuneKar misal Corner": (6.0, 4.0, 6.5, 9.0),
        "Swami Hotel": (9.5, 9.5, 9.5, 4.0),
        "McDonald's": (5.5, 3.0, 7.0, 8.5),
        "Pizza Hut": (8.0, 9.5, 8.0, 7.0),
        "KFC": (6.5, 5.0, 6.0, 7.5),
        "Domino's": (9.0, 8.5, 8.5, 5.0)
    }
    
    
    excellent_restaurants = {}
    good_value_restaurants = {}
    authentic_restaurants = {}
    
    for restaurant, (rating, authenticity, service, value) in restaurants.items():
        # Excellent restaurants (high overall quality)
        excellent_membership = restaurant_membership_function(rating, authenticity, service, 5.0)
        if excellent_membership > 0.3:
            excellent_restaurants[restaurant] = excellent_membership
        
        # Good value restaurants (emphasis on price value)
        value_membership = restaurant_membership_function(rating, 5.0, service, value)
        if value_membership > 0.3:
            good_value_restaurants[restaurant] = value_membership
            
        # Authentic restaurants (emphasis on cuisine authenticity)
        authentic_membership = restaurant_membership_function(rating, authenticity, 6.0, 6.0)
        if authentic_membership > 0.3:
            authentic_restaurants[restaurant] = authentic_membership
    
   
    excellent_set = FuzzySet("Excellent Restaurants", excellent_restaurants)
    value_set = FuzzySet("Good Value Restaurants", good_value_restaurants)
    authentic_set = FuzzySet("Authentic Restaurants", authentic_restaurants)
    
    print("\n1. ORIGINAL FUZZY SETS:")
    print(f"   {excellent_set}")
    print(f"   {value_set}")
    print(f"   {authentic_set}")
    
  
    print("\n2. FUZZY SET OPERATIONS:")
    
    # Union: Restaurants that are either excellent OR good value
    union_result = excellent_set.union(value_set)
    print(f"\n   Union (Excellent ∪ Good Value):")
    print(f"   {union_result}")
    
    # Intersection: Restaurants that are both excellent AND good value
    intersection_result = excellent_set.intersection(value_set)
    print(f"\n   Intersection (Excellent ∩ Good Value):")
    print(f"   {intersection_result}")
    
    # Complement: Restaurants that are NOT excellent
    complement_result = excellent_set.complement()
    print(f"\n   Complement (¬Excellent):")
    print(f"   {complement_result}")
    
    # Difference: Excellent restaurants that are NOT good value
    difference_result = excellent_set.difference(value_set)
    print(f"\n   Difference (Excellent - Good Value):")
    print(f"   {difference_result}")





def main():
    

    assignment_1_demo()
    
  
if __name__ == "__main__":
    main()














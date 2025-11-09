# =============================================================================
# ASSIGNMENT 2: RELATIONAL FUZZY OPERATIONS
# Real-life Example: Student-Course Matching System
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd

class FuzzyRelation:
    """Class to represent fuzzy relations"""
    
    def __init__(self, name: str, matrix: np.ndarray, row_labels: List[str], col_labels: List[str]):
        self.name = name
        self.matrix = np.clip(matrix, 0, 1) 
        self.row_labels = row_labels
        self.col_labels = col_labels
    
    def __str__(self):
        df = pd.DataFrame(self.matrix, index=self.row_labels, columns=self.col_labels)
        return f"{self.name}:\n{df}"
    
    def composition(self, other):
        """Max-min composition of two fuzzy relations"""
        if len(self.col_labels) != len(other.row_labels):
            raise ValueError("Incompatible dimensions for composition")
        
        result_matrix = np.zeros((len(self.row_labels), len(other.col_labels)))
        
        for i in range(len(self.row_labels)):
            for j in range(len(other.col_labels)):
                # Max-min composition
                values = []
                for k in range(len(self.col_labels)):
                    values.append(min(self.matrix[i][k], other.matrix[k][j]))
                result_matrix[i][j] = max(values)
        
        return FuzzyRelation(
            f"{self.name} ∘ {other.name}",
            result_matrix,
            self.row_labels,
            other.col_labels
        )
    
    def transpose(self):
        """Transpose of fuzzy relation"""
        return FuzzyRelation(
            f"{self.name}ᵀ",
            self.matrix.T,
            self.col_labels,
            self.row_labels
        )
    
    def union(self, other):
        """Union of two fuzzy relations"""
        if self.matrix.shape != other.matrix.shape:
            raise ValueError("Relations must have same dimensions")
        
        result_matrix = np.maximum(self.matrix, other.matrix)
        return FuzzyRelation(
            f"{self.name} ∪ {other.name}",
            result_matrix,
            self.row_labels,
            self.col_labels
        )
    
    def intersection(self, other):
        """Intersection of two fuzzy relations"""
        if self.matrix.shape != other.matrix.shape:
            raise ValueError("Relations must have same dimensions")
        
        result_matrix = np.minimum(self.matrix, other.matrix)
        return FuzzyRelation(
            f"{self.name} ∩ {other.name}",
            result_matrix,
            self.row_labels,
            self.col_labels
        )

def assignment_2_demo():
    
    
    # Students and their skill levels in different areas
    students = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    skills = ["Math", "Programming", "Communication", "Creativity", "Analytics"]
    
    # Student-Skill relation (how good each student is at each skill)
    student_skill_matrix = np.array([
        [0.9, 0.8, 0.7, 0.6, 0.8],  # Alice
        [0.7, 0.9, 0.5, 0.4, 0.7],  # Bob
        [0.6, 0.6, 0.9, 0.8, 0.6],  # Charlie
        [0.8, 0.5, 0.8, 0.9, 0.7],  # Diana
        [0.9, 0.7, 0.6, 0.5, 0.9]   # Eve
    ])
    
    courses = ["Data Science", "Web Development", "Digital Marketing", "Game Design", "Business Analytics"]
    
    # Skill-Course relation (how much each skill is needed for each course)
    skill_course_matrix = np.array([
        [0.9, 0.5, 0.3, 0.4, 0.8],  # Math
        [0.8, 0.9, 0.4, 0.8, 0.7],  # Programming
        [0.6, 0.6, 0.9, 0.7, 0.8],  # Communication
        [0.4, 0.7, 0.8, 0.9, 0.5],  # Creativity
        [0.9, 0.4, 0.7, 0.3, 0.9]   # Analytics
    ])
    
    # Create fuzzy relations
    student_skill_rel = FuzzyRelation("Student-Skill", student_skill_matrix, students, skills)
    skill_course_rel = FuzzyRelation("Skill-Course", skill_course_matrix, skills, courses)
    
    print("\n1. ORIGINAL FUZZY RELATIONS:")
    print(f"\n{student_skill_rel}")
    print(f"\n{skill_course_rel}")
    
    print("\n2. RELATIONAL OPERATIONS:")
    
    # Composition: Student-Course suitability
    student_course_rel = student_skill_rel.composition(skill_course_rel)
    print(f"\n   Composition (Student-Course Suitability):")
    print(f"   {student_course_rel}")
    
    # Transpose
    transposed_rel = student_skill_rel.transpose()
    print(f"\n   Transpose of Student-Skill relation:")
    print(f"   {transposed_rel}")
    

    # Student preferences for different course types
    course_types = ["Technical", "Creative", "Business", "Research", "Practical"]
    student_preference_matrix = np.array([
        [0.8, 0.6, 0.7, 0.9, 0.7],  # Alice
        [0.9, 0.4, 0.6, 0.8, 0.8],  # Bob
        [0.5, 0.9, 0.8, 0.6, 0.7],  # Charlie
        [0.6, 0.8, 0.9, 0.7, 0.6],  # Diana
        [0.9, 0.5, 0.8, 0.9, 0.8]   # Eve
    ])
    
    student_preference_rel = FuzzyRelation("Student-Preference", student_preference_matrix, students, course_types)
    

    modified_skill_matrix = student_skill_matrix  #
    modified_skill_rel = FuzzyRelation("Modified Student-Skill", modified_skill_matrix, students, skills)
    
    print(f"\n   Modified Student-Skill relation for set operations:")
    print(f"   {modified_skill_rel}")







class MembershipFunctions:
    """Collection of different membership functions with parameter analysis"""
    
    @staticmethod
    def triangular(x, a, b, c):
        """Triangular membership function"""
        return np.maximum(0, np.minimum((x - a) / (b - a), (c - x) / (c - b)))
    
    @staticmethod
    def trapezoidal(x, a, b, c, d):
        """Trapezoidal membership function"""
        return np.maximum(0, np.minimum(np.minimum((x - a) / (b - a), 1), (d - x) / (d - c)))
    
    @staticmethod
    def gaussian(x, mean, std):
        """Gaussian membership function"""
        return np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    @staticmethod
    def sigmoid(x, a, c):
        """Sigmoid membership function"""
        return 1 / (1 + np.exp(-a * (x - c)))
    
    @staticmethod
    def bell(x, a, b, c):
        """Bell-shaped membership function"""
        return 1 / (1 + np.abs((x - c) / a) ** (2 * b))

def plot_membership_function_analysis():
    
    # Temperature range for analysis (in Celsius)
    x = np.linspace(0, 40, 400)
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Membership Function Analysis: Smart Home Temperature Control', fontsize=16, fontweight='bold')
    
    # 1. TRIANGULAR MEMBERSHIP FUNCTION
    print("\n1. TRIANGULAR MEMBERSHIP FUNCTION ANALYSIS:")
    print("   Parameters: a (left base), b (peak), c (right base)")
    
    ax1 = axes[0, 0]
    # Vary parameter 'b' (peak position)
    for b in [10, 15, 20, 25]:
        y = MembershipFunctions.triangular(x, 5, b, 35)
        ax1.plot(x, y, label=f'Peak at {b}°C', linewidth=2)
    ax1.set_title('Effect of Peak Position (b)', fontweight='bold')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Membership Degree')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    print("   ANALYSIS: Changing peak position (b) shifts the temperature where membership is highest.")
    print("   COMMENT: In temperature control, this represents the ideal comfortable temperature.")
    print("   PRACTICAL IMPACT: Moving peak from 15°C to 25°C changes 'comfortable' from cool to warm.")
    
    ax2 = axes[0, 1]
    # Vary parameter 'a' and 'c' (base width)
    base_configs = [(5, 20, 35), (10, 20, 30), (0, 20, 40), (15, 20, 25)]
    for a, b, c in base_configs:
        y = MembershipFunctions.triangular(x, a, b, c)
        width = c - a
        ax2.plot(x, y, label=f'Width: {width}°C', linewidth=2)
    ax2.set_title('Effect of Base Width (c-a)', fontweight='bold')
    ax2.set_xlabel('Temperature (°C)')
    ax2.set_ylabel('Membership Degree')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    print("   ANALYSIS: Changing base width affects how tolerant the system is to temperature variations.")
    print("   COMMENT: Narrow base = strict temperature control, Wide base = more flexible control.")
    
    # 2. TRAPEZOIDAL MEMBERSHIP FUNCTION
    print("\n2. TRAPEZOIDAL MEMBERSHIP FUNCTION ANALYSIS:")
    print("   Parameters: a (left base), b (left top), c (right top), d (right base)")
    
    ax3 = axes[1, 0]
    # Vary flat top width
    for b, c in [(15, 20), (12, 23), (10, 25), (8, 27)]:
        y = MembershipFunctions.trapezoidal(x, 5, b, c, 35)
        flat_width = c - b
        ax3.plot(x, y, label=f'Flat width: {flat_width}°C', linewidth=2)
    ax3.set_title('Effect of Flat Top Width (c-b)', fontweight='bold')
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Membership Degree')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    print("   ANALYSIS: Flat top width determines the range of temperatures considered equally comfortable.")
    print("   COMMENT: Wider flat top = larger comfort zone, Narrower = more precise comfort requirement.")
    print("   PRACTICAL IMPACT: 2°C flat zone vs 10°C zone affects heating/cooling frequency.")
    
    ax4 = axes[1, 1]
    # Vary slope steepness
    slope_configs = [(5, 15, 25, 35), (10, 17, 23, 30), (0, 12, 28, 40), (8, 18, 22, 32)]
    for a, b, c, d in slope_configs:
        y = MembershipFunctions.trapezoidal(x, a, b, c, d)
        slope = 1 / (b - a) if b != a else float('inf')
        ax4.plot(x, y, label=f'Slope ≈ {slope:.2f}', linewidth=2)
    ax4.set_title('Effect of Slope Steepness', fontweight='bold')
    ax4.set_xlabel('Temperature (°C)')
    ax4.set_ylabel('Membership Degree')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    print("   ANALYSIS: Slope steepness affects how quickly membership drops outside comfort zone.")
    print("   COMMENT: Steep slopes = sharp transitions, Gentle slopes = gradual transitions.")
    
    # 3. GAUSSIAN MEMBERSHIP FUNCTION
    print("\n3. GAUSSIAN MEMBERSHIP FUNCTION ANALYSIS:")
    print("   Parameters: mean (center), std (standard deviation)")
    
    ax5 = axes[2, 0]
    # Vary mean
    for mean in [15, 20, 25, 30]:
        y = MembershipFunctions.gaussian(x, mean, 5)
        ax5.plot(x, y, label=f'Mean: {mean}°C', linewidth=2)
    ax5.set_title('Effect of Mean (Center Position)', fontweight='bold')
    ax5.set_xlabel('Temperature (°C)')
    ax5.set_ylabel('Membership Degree')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    print("   ANALYSIS: Mean shifts the center of the bell curve (optimal temperature).")
    print("   COMMENT: Similar to triangular peak, but with smooth, natural distribution.")
    print("   PRACTICAL IMPACT: Gaussian curves model human comfort perception more naturally.")
    
    ax6 = axes[2, 1]
    # Vary standard deviation
    for std in [2, 4, 6, 8]:
        y = MembershipFunctions.gaussian(x, 22, std)
        ax6.plot(x, y, label=f'Std Dev: {std}°C', linewidth=2)
    ax6.set_title('Effect of Standard Deviation (Spread)', fontweight='bold')
    ax6.set_xlabel('Temperature (°C)')
    ax6.set_ylabel('Membership Degree')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    print("   ANALYSIS: Standard deviation controls the width/spread of the membership function.")
    print("   COMMENT: Small std = narrow comfort zone, Large std = wide tolerance.")
    print("   PRACTICAL IMPACT: Std=2°C requires precise control, Std=8°C allows more variation.")
    
    plt.tight_layout()
    plt.show()
    
    # Additional analysis plots
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle('Advanced Membership Functions: Sigmoid and Bell', fontsize=14, fontweight='bold')
    
    # 4. SIGMOID MEMBERSHIP FUNCTION
    print("\n4. SIGMOID MEMBERSHIP FUNCTION ANALYSIS:")
    print("   Parameters: a (slope steepness), c (inflection point)")
    
    ax7 = axes2[0, 0]
    # Vary slope parameter 'a'
    for a in [0.2, 0.5, 1.0, 2.0]:
        y = MembershipFunctions.sigmoid(x, a, 20)
        ax7.plot(x, y, label=f'Slope a={a}', linewidth=2)
    ax7.set_title('Sigmoid: Effect of Slope Parameter (a)', fontweight='bold')
    ax7.set_xlabel('Temperature (°C)')
    ax7.set_ylabel('Membership Degree')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    print("   ANALYSIS: Parameter 'a' controls how sharp the transition is.")
    print("   COMMENT: Low 'a' = gradual transition, High 'a' = sharp transition.")
    print("   PRACTICAL IMPACT: Represents 'too hot' threshold - sharp vs gradual onset.")
    
    ax8 = axes2[0, 1]
    # Vary inflection point 'c'
    for c in [15, 20, 25, 30]:
        y = MembershipFunctions.sigmoid(x, 0.5, c)
        ax8.plot(x, y, label=f'Inflection at {c}°C', linewidth=2)
    ax8.set_title('Sigmoid: Effect of Inflection Point (c)', fontweight='bold')
    ax8.set_xlabel('Temperature (°C)')
    ax8.set_ylabel('Membership Degree')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    print("   ANALYSIS: Parameter 'c' shifts the S-curve left or right.")
    print("   COMMENT: Determines the temperature threshold for the transition.")
    
    # 5. BELL-SHAPED MEMBERSHIP FUNCTION
    print("\n5. BELL-SHAPED MEMBERSHIP FUNCTION ANALYSIS:")
    print("   Parameters: a (width), b (slope), c (center)")
    
    ax9 = axes2[1, 0]
    # Vary width parameter 'a'
    for a in [3, 5, 8, 12]:
        y = MembershipFunctions.bell(x, a, 2, 22)
        ax9.plot(x, y, label=f'Width a={a}', linewidth=2)
    ax9.set_title('Bell: Effect of Width Parameter (a)', fontweight='bold')
    ax9.set_xlabel('Temperature (°C)')
    ax9.set_ylabel('Membership Degree')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    print("   ANALYSIS: Parameter 'a' controls the width of the bell curve.")
    print("   COMMENT: Larger 'a' = wider bell, more tolerance to temperature variation.")
    print("   PRACTICAL IMPACT: Wide bell allows more temperature fluctuation before discomfort.")
    
    ax10 = axes2[1, 1]
    # Vary slope parameter 'b'
    for b in [1, 2, 4, 8]:
        y = MembershipFunctions.bell(x, 6, b, 22)
        ax10.plot(x, y, label=f'Slope b={b}', linewidth=2)
    ax10.set_title('Bell: Effect of Slope Parameter (b)', fontweight='bold')
    ax10.set_xlabel('Temperature (°C)')
    ax10.set_ylabel('Membership Degree')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    
    print("   ANALYSIS: Parameter 'b' controls the steepness of the bell curve sides.")
    print("   COMMENT: Higher 'b' = steeper sides, more concentrated membership around center.")
    print("   PRACTICAL IMPACT: Steep sides mean rapid comfort loss outside optimal range.")
    
    plt.tight_layout()
    plt.show()
    
    # COMPREHENSIVE COMPARISON
    print("\n" + "="*60)
    print("COMPREHENSIVE MEMBERSHIP FUNCTION COMPARISON")
    print("="*60)
    
    fig3, ax = plt.subplots(figsize=(12, 8))
    
    # Compare all functions with similar parameters
    y_triangular = MembershipFunctions.triangular(x, 10, 22, 34)
    y_trapezoidal = MembershipFunctions.trapezoidal(x, 10, 18, 26, 34)
    y_gaussian = MembershipFunctions.gaussian(x, 22, 4)
    y_sigmoid = MembershipFunctions.sigmoid(x, 0.3, 22)
    y_bell = MembershipFunctions.bell(x, 6, 2, 22)
    
    ax.plot(x, y_triangular, label='Triangular', linewidth=3, alpha=0.8)
    ax.plot(x, y_trapezoidal, label='Trapezoidal', linewidth=3, alpha=0.8)
    ax.plot(x, y_gaussian, label='Gaussian', linewidth=3, alpha=0.8)
    ax.plot(x, y_sigmoid, label='Sigmoid', linewidth=3, alpha=0.8)
    ax.plot(x, y_bell, label='Bell-shaped', linewidth=3, alpha=0.8)
    
    ax.set_title('Membership Functions Comparison: Temperature Control', fontsize=14, fontweight='bold')
    ax.set_xlabel('Temperature (°C)', fontsize=12)
    ax.set_ylabel('Membership Degree', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    
    plt.tight_layout()
    plt.show()
    
    print("\nFINAL ANALYSIS SUMMARY:")
    print("1. TRIANGULAR: Simple, computationally efficient, good for basic control")
    print("2. TRAPEZOIDAL: Provides comfort zones, good for applications needing stable regions")
    print("3. GAUSSIAN: Natural, smooth transitions, models human perception well")
    print("4. SIGMOID: Excellent for threshold-based decisions, asymmetric behavior")
    print("5. BELL-SHAPED: Combines smoothness with controllable steepness")
    
    print("\nRECOMMENDATIONS FOR SMART HOME TEMPERATURE CONTROL:")
    print("- Use GAUSSIAN for comfort modeling (natural human response)")
    print("- Use TRAPEZOIDAL for energy-efficient control (stable zones)")
    print("- Use SIGMOID for emergency responses (sharp thresholds)")
    print("- Use TRIANGULAR for simple, fast processing applications")
    print("- Use BELL-SHAPED for premium systems requiring fine-tuned control")

def main():
    

    assignment_2_demo()
    plot_membership_function_analysis()
    
  
if __name__ == "__main__":
    main()
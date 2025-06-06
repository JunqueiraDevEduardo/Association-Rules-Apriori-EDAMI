############################################
#               STEP 3: association_rules.py 
# Association Rules Generation from Frequent Itemsets
# This file implements the association rules generation from frequent itemsets
# found in Step 2, calculating confidence and other metrics
# EDUARDO JUNQUEIRA 
# TOPIC 1 ASSOCIATION RULES WITH @APRIORI ALGORITHM PROJECT Data Mining
############################################

############################################
# Import libraries
############################################
import time  # For measuring execution time
from itertools import combinations  # For generating rule combinations
from apriori_core import AprioriAlgorithm  # Import Step 2 class

############################################
# Class: AssociationRulesGenerator
# Generate association rules from frequent itemsets
############################################

class AssociationRulesGenerator:
    def __init__(self, verbose=True):
        """
        Initialize Association Rules Generator
        
        Args:
            verbose (bool): Print processing information
        """
        self.verbose = verbose
        self.apriori = AprioriAlgorithm(verbose=False)  # Initialize Apriori algorithm
        self.frequent_itemsets = {}  # Store frequent itemsets from Step 2
        self.association_rules = []  # Store generated association rules
        self.transactions = []  # Store transactions for support calculation
        
    def load_frequent_itemsets(self, limit=50, min_support=0.05):
        """
        Load frequent itemsets using Step 2 (Apriori Core)
        
        Args:
            limit (int): Number of transactions to process
            min_support (float): Minimum support threshold
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"LOADING FREQUENT ITEMSETS FROM STEP 2")
            print(f"{'='*80}")
            
        # Load transactions and find frequent itemsets
        transaction_count = self.apriori.load_transactions(limit=limit)
        self.transactions = self.apriori.transactions
        
        if transaction_count == 0:
            print("Error: No transactions loaded!")
            return False
            
        # Find frequent itemsets using Apriori algorithm
        self.frequent_itemsets = self.apriori.find_frequent_itemsets(min_support)
        
        if self.verbose:
            total_itemsets = sum(len(itemsets) for itemsets in self.frequent_itemsets.values())
            print(f"Loaded {total_itemsets} frequent itemsets for rule generation")
            
        return True
        
    def calculate_support(self, itemset):
        """
        Calculate support for an itemset
        
        Args:
            itemset (set/frozenset): Items to calculate support for
            
        Returns:
            float: Support value (0.0 to 1.0)
        """
        if not self.transactions:
            return 0.0
            
        itemset_set = set(itemset) if not isinstance(itemset, set) else itemset
        count = sum(1 for transaction in self.transactions if itemset_set.issubset(set(transaction)))
        return count / len(self.transactions)
        
    def calculate_confidence(self, antecedent, consequent):
        """
        Calculate confidence for a rule: antecedent -> consequent
        Confidence = Support(antecedent ∪ consequent) / Support(antecedent)
        
        Args:
            antecedent (frozenset): Left side of the rule
            consequent (frozenset): Right side of the rule
            
        Returns:
            float: Confidence value (0.0 to 1.0)
        """
        # Union of antecedent and consequent
        union_set = antecedent.union(consequent)
        
        # Calculate supports
        support_union = self.calculate_support(union_set)
        support_antecedent = self.calculate_support(antecedent)
        
        # Avoid division by zero
        if support_antecedent == 0:
            return 0.0
            
        confidence = support_union / support_antecedent
        return confidence
        
    def calculate_lift(self, antecedent, consequent):
        """
        Calculate lift for a rule: antecedent -> consequent
        Lift = Confidence(antecedent -> consequent) / Support(consequent)
        
        Args:
            antecedent (frozenset): Left side of the rule
            consequent (frozenset): Right side of the rule
            
        Returns:
            float: Lift value
        """
        confidence = self.calculate_confidence(antecedent, consequent)
        support_consequent = self.calculate_support(consequent)
        
        # Avoid division by zero
        if support_consequent == 0:
            return 0.0
            
        lift = confidence / support_consequent
        return lift
        
    def generate_candidates_from_itemset(self, itemset):
        """
        Generate all possible rule candidates from a frequent itemset
        For itemset {A, B, C}, generate: A->BC, B->AC, C->AB, AB->C, AC->B, BC->A
        
        Args:
            itemset (frozenset): Frequent itemset to generate rules from
            
        Returns:
            list: List of (antecedent, consequent) tuples
        """
        candidates = []
        items = list(itemset)
        
        # Generate all possible antecedent sizes (1 to len(itemset)-1)
        for ant_size in range(1, len(items)):
            # Generate all combinations of antecedent_size items
            for antecedent_items in combinations(items, ant_size):
                antecedent = frozenset(antecedent_items)
                consequent = itemset - antecedent  # Remaining items
                candidates.append((antecedent, consequent))
                
        return candidates
        
    def generate_association_rules(self, min_confidence=0.5):
        """
        Generate association rules from frequent itemsets
        
        Args:
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            list: List of association rules with metrics
        """
        if not self.frequent_itemsets:
            print("Error: No frequent itemsets available. Run load_frequent_itemsets() first!")
            return []
            
        start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"GENERATING ASSOCIATION RULES")
            print(f"{'='*80}")
            print(f"Minimum confidence threshold: {min_confidence}")
            
        self.association_rules = []
        rule_count = 0
        
        # Process frequent itemsets of size 2 and above
        for k, itemsets in self.frequent_itemsets.items():
            if k < 2:  # Skip 1-itemsets (can't make rules)
                continue
                
            if self.verbose:
                print(f"\n{'-'*60}")
                print(f"PROCESSING {k}-ITEMSETS FOR RULE GENERATION")
                print(f"{'-'*60}")
                print(f"Processing {len(itemsets)} frequent {k}-itemsets...")
            
            for itemset in itemsets:
                # Generate all possible rule candidates from this itemset
                candidates = self.generate_candidates_from_itemset(itemset)
                
                for antecedent, consequent in candidates:
                    # Calculate metrics
                    confidence = self.calculate_confidence(antecedent, consequent)
                    
                    # Check if rule meets minimum confidence
                    if confidence >= min_confidence:
                        support = self.calculate_support(itemset)
                        lift = self.calculate_lift(antecedent, consequent)
                        
                        # Create rule dictionary
                        rule = {
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': support,
                            'confidence': confidence,
                            'lift': lift,
                            'antecedent_support': self.calculate_support(antecedent),
                            'consequent_support': self.calculate_support(consequent)
                        }
                        
                        self.association_rules.append(rule)
                        rule_count += 1
        
        # Sort rules by confidence (descending)
        self.association_rules.sort(key=lambda x: x['confidence'], reverse=True)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"ASSOCIATION RULES GENERATION RESULTS")
            print(f"{'='*80}")
            print(f"Processing time: {processing_time:.2f} seconds")
            print(f"Total rules generated: {len(self.association_rules)}")
            print(f"Rules meeting confidence threshold ({min_confidence}): {rule_count}")
            
        return self.association_rules
        
    def display_rules(self, max_rules=20, sort_by='confidence'):
        """
        Display association rules in a formatted way
        
        Args:
            max_rules (int): Maximum number of rules to display
            sort_by (str): Sort criterion ('confidence', 'support', 'lift')
        """
        if not self.association_rules:
            print("No association rules to display!")
            return
            
        # Sort rules based on the specified criterion
        if sort_by == 'support':
            sorted_rules = sorted(self.association_rules, key=lambda x: x['support'], reverse=True)
        elif sort_by == 'lift':
            sorted_rules = sorted(self.association_rules, key=lambda x: x['lift'], reverse=True)
        else:  # Default to confidence
            sorted_rules = sorted(self.association_rules, key=lambda x: x['confidence'], reverse=True)
            
        print(f"\n{'='*120}")
        print(f"TOP {min(max_rules, len(sorted_rules))} ASSOCIATION RULES (sorted by {sort_by.upper()})")
        print(f"{'='*120}")
        
        # Header
        print(f"{'#':<3} {'RULE':<50} {'SUPP':<8} {'CONF':<8} {'LIFT':<8}")
        print(f"{'-'*120}")
        
        # Display rules
        for i, rule in enumerate(sorted_rules[:max_rules]):
            antecedent_str = ', '.join(sorted(list(rule['antecedent'])))
            consequent_str = ', '.join(sorted(list(rule['consequent'])))
            
            # Truncate long item names
            if len(antecedent_str) > 20:
                antecedent_str = antecedent_str[:17] + "..."
            if len(consequent_str) > 20:
                consequent_str = consequent_str[:17] + "..."
                
            rule_str = f"{{{antecedent_str}}} -> {{{consequent_str}}}"
            
            print(f"{i+1:<3} {rule_str:<50} {rule['support']:<8.4f} {rule['confidence']:<8.4f} {rule['lift']:<8.4f}")
            
    def analyze_rules(self):
        """
        Provide statistical analysis of the generated rules
        """
        if not self.association_rules:
            print("No rules to analyze!")
            return
            
        print(f"\n{'='*80}")
        print(f"ASSOCIATION RULES ANALYSIS")
        print(f"{'='*80}")
        
        # Basic statistics
        confidences = [rule['confidence'] for rule in self.association_rules]
        supports = [rule['support'] for rule in self.association_rules]
        lifts = [rule['lift'] for rule in self.association_rules]
        
        print(f"Total rules: {len(self.association_rules)}")
        print(f"\nConfidence statistics:")
        print(f"  Average: {sum(confidences)/len(confidences):.4f}")
        print(f"  Minimum: {min(confidences):.4f}")
        print(f"  Maximum: {max(confidences):.4f}")
        
        print(f"\nSupport statistics:")
        print(f"  Average: {sum(supports)/len(supports):.4f}")
        print(f"  Minimum: {min(supports):.4f}")
        print(f"  Maximum: {max(supports):.4f}")
        
        print(f"\nLift statistics:")
        print(f"  Average: {sum(lifts)/len(lifts):.4f}")
        print(f"  Minimum: {min(lifts):.4f}")
        print(f"  Maximum: {max(lifts):.4f}")
        
        # Count rules by lift interpretation
        strong_positive = sum(1 for lift in lifts if lift > 1.5)
        moderate_positive = sum(1 for lift in lifts if 1.1 <= lift <= 1.5)
        weak_positive = sum(1 for lift in lifts if 1.0 < lift < 1.1)
        independent = sum(1 for lift in lifts if lift == 1.0)
        negative = sum(1 for lift in lifts if lift < 1.0)
        
        print(f"\nLift interpretation:")
        print(f"  Strong positive correlation (lift > 1.5): {strong_positive}")
        print(f"  Moderate positive correlation (1.1 ≤ lift ≤ 1.5): {moderate_positive}")
        print(f"  Weak positive correlation (1.0 < lift < 1.1): {weak_positive}")
        print(f"  Independent (lift = 1.0): {independent}")
        print(f"  Negative correlation (lift < 1.0): {negative}")
        
    def run_multiple_confidence_thresholds(self, confidence_thresholds=[0.8, 0.6, 0.4]):
        """
        Run rule generation with multiple confidence thresholds
        
        Args:
            confidence_thresholds (list): List of confidence thresholds to test
        """
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"TESTING MULTIPLE CONFIDENCE THRESHOLDS")
            print(f"{'='*80}")
        
        results = {}
        
        for threshold in confidence_thresholds:
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"TESTING MINIMUM CONFIDENCE: {threshold}")
                print(f"{'='*60}")
            
            rules = self.generate_association_rules(min_confidence=threshold)
            results[threshold] = rules
            
            if self.verbose:
                print(f"Rules with confidence >= {threshold}: {len(rules)}")
                
        return results

############################################
# Main execution function
############################################

def main():
    """
    Main function to run Step 3 of the Apriori algorithm
    """
    print("="*80)
    print("@APRIORI ALGORITHM PROJECT - ASSOCIATION RULES MINING")
    print("EDUARDO JUNQUEIRA - @EDAMI")
    print("="*80)
    print("STEP 3: ASSOCIATION RULES GENERATION")
    print("="*80)
    
    # Initialize Association Rules Generator
    rules_generator = AssociationRulesGenerator(verbose=True)
    
    # Load frequent itemsets from Step 2
    success = rules_generator.load_frequent_itemsets(limit=50, min_support=0.05)
    
    if not success:
        print("Error: Failed to load frequent itemsets. Please check Steps 1 and 2.")
        return
    
    # Generate association rules with different confidence thresholds
    confidence_thresholds = [0.8, 0.6, 0.4]
    results = rules_generator.run_multiple_confidence_thresholds(confidence_thresholds)
    
    # Display results for the most lenient threshold
    if results:
        best_threshold = min(confidence_thresholds)  # Most lenient threshold
        rules_generator.association_rules = results[best_threshold]
        
        print(f"\n{'='*80}")
        print(f"DISPLAYING RESULTS FOR CONFIDENCE THRESHOLD: {best_threshold}")
        print(f"{'='*80}")
        
        # Display top rules
        rules_generator.display_rules(max_rules=15, sort_by='confidence')
        
        # Analyze rules
        rules_generator.analyze_rules()
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY - STEP 3 COMPLETED")
    print(f"{'='*80}")
    
    for threshold, rules in results.items():
        print(f"Confidence {threshold}: {len(rules)} association rules")
    
    print(f"\nStep 3 completed successfully!")
    print(f"Association rules mining project completed!")

############################################
# Execute main function if script is run directly
############################################

if __name__ == "__main__":
    main()